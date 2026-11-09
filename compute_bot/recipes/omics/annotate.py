"""Network annotation: UniProt (protein name, function, GO, location) and g:Profiler enrichment.

Both take a `fetch(url, data=None, timeout=30) -> bytes` so tests can mock it, cache answers in a JSON file next to
the results (a rerun or an offline rerun gives the same tables), and never raise on network trouble: they return
what they have plus a warning for the summary. COMPUTE_BOT_OFFLINE=1 makes the default fetch fail at once.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

UNIPROT = "https://rest.uniprot.org/uniprotkb"
GPROFILER = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
FIELDS = ["accession", "protein_name", "gene_primary", "cc_function", "go_p", "go_f", "go_c",
          "cc_subcellular_location"]
COLUMNS = ["accession", "protein_name", "gene", "function", "go_bp", "go_mf", "go_cc", "location"]
TAXON = {"hsapiens": 9606, "mmusculus": 10090, "rnorvegicus": 10116, "scerevisiae": 559292, "ecoli": 83333,
         "dmelanogaster": 7227, "celegans": 6239, "drerio": 7955, "athaliana": 3702, "btaurus": 9913,
         "sscrofa": 9823, "ggallus": 9031, "oaries": 9940, "ocuniculus": 9986, "cfamiliaris": 9615}
SOURCES = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC", "WP"]


class Offline(OSError):
    pass


def http_fetch(url, data=None, timeout=30):
    if os.environ.get("COMPUTE_BOT_OFFLINE"):
        raise Offline("brak sieci (COMPUTE_BOT_OFFLINE)")
    headers = {"User-Agent": "compute-bot omics recipe"}
    if data is not None:
        headers["Content-Type"] = "application/json"
        data = data.encode() if isinstance(data, str) else data
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def _save(path, cache):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, sort_keys=True, ensure_ascii=False))
    tmp.replace(path)


def _clean(text, prefix):
    t = re.sub(r"\s*\((PubMed|ECO):[^)]*\)", "", str(text or ""))
    t = re.sub(r"\{ECO:[^}]*\}", "", t)
    return t.replace(prefix, "").strip(" .;") or ""


def _parse_tsv(raw):
    rows = []
    lines = raw.decode("utf-8", errors="replace").splitlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = (line.split("\t") + [""] * len(FIELDS))[: len(FIELDS)]
        rec = dict(zip(COLUMNS, parts, strict=True))
        rec["gene"] = rec["gene"].split(";")[0].strip()
        rec["function"] = _clean(rec["function"], "FUNCTION:")
        rec["location"] = _clean(rec["location"], "SUBCELLULAR LOCATION:")
        rows.append(rec)
    return rows


def _why(e):
    return str(e) if isinstance(e, Offline) else f"{e.__class__.__name__}: {str(e)[:80]}"


def uniprot(accessions=None, genes=None, organism="hsapiens", cache_path="uniprot_cache.json", fetch=http_fetch,
            batch=100, timeout=30):
    """Annotation table indexed by the query (accession or gene) for the entries UniProt knows."""
    cache = _load(cache_path)
    taxon = TAXON.get(str(organism).lower())
    warn = None
    if accessions:
        queries = [(a, "acc:" + re.sub(r"-\d+$", "", a)) for a in dict.fromkeys(accessions) if a]
    else:
        queries = [(g, f"gene:{taxon}:{g.upper()}") for g in dict.fromkeys(genes or []) if g]
    todo = [q for q in queries if q[1] not in cache]
    try:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            if accessions:
                ids = [k[4:] for _, k in chunk]
                url = f"{UNIPROT}/accessions?" + urllib.parse.urlencode(
                    {"accessions": ",".join(ids), "fields": ",".join(FIELDS), "format": "tsv"})
                rows = _parse_tsv(fetch(url, timeout=timeout))
                by = {r["accession"]: r for r in rows}
                for _, k in chunk:
                    cache[k] = by.get(k[4:])
            else:
                names = [q for q, _ in chunk]
                query = "(" + " OR ".join(f"gene_exact:{g}" for g in names) + ")"
                if taxon:
                    query += f" AND organism_id:{taxon}"
                query += " AND reviewed:true"
                url = f"{UNIPROT}/search?" + urllib.parse.urlencode(
                    {"query": query, "fields": ",".join(FIELDS), "format": "tsv", "size": 500})
                rows = _parse_tsv(fetch(url, timeout=timeout))
                by = {}
                for r in rows:
                    by.setdefault(r["gene"].upper(), r)
                for g, k in chunk:
                    cache[k] = by.get(g.upper())
            _save(cache_path, cache)
    except Exception as e:  # noqa: BLE001 - any network/HTTP failure only skips the optional step
        warn = f"UniProt niedostępny ({_why(e)}) – adnotacja pominięta"
        if any(cache.get(k) for _, k in queries):
            warn += " (część z pamięci podręcznej)"
    recs = {q: cache[k] for q, k in queries if cache.get(k)}
    tab = pd.DataFrame.from_dict(recs, orient="index", columns=COLUMNS) if recs else pd.DataFrame(columns=COLUMNS)
    return tab, warn


def gprofiler(queries, background=None, organism="hsapiens", cache_path="gprofiler_cache.json", fetch=http_fetch,
              sources=None, timeout=60):
    """g:GOSt enrichment of each gene list vs the measured background (g:SCS threshold 0.05)."""
    queries = {k: list(dict.fromkeys(v)) for k, v in queries.items() if len(set(v)) >= 3}
    cols = ["query", "source", "native", "name", "p_value", "term_size", "query_size", "intersection_size"]
    if not queries:
        return pd.DataFrame(columns=cols), None
    body = {"organism": organism, "query": queries, "sources": sources or SOURCES, "user_threshold": 0.05,
            "significance_threshold_method": "g_SCS", "no_evidences": True, "all_results": False,
            "ordered": False}
    if background:
        body.update(domain_scope="custom", background=list(background))
    text = json.dumps(body, sort_keys=True)
    key = hashlib.sha1(text.encode()).hexdigest()
    cache = _load(cache_path)
    warn = None
    if key not in cache:
        try:
            cache[key] = json.loads(fetch(GPROFILER, data=text, timeout=timeout)).get("result", [])
            _save(cache_path, cache)
        except Exception as e:  # noqa: BLE001 - optional step: report and go on
            return pd.DataFrame(columns=cols), f"g:Profiler niedostępny ({_why(e)}) – wzbogacenie pominięte"
    res = pd.DataFrame(cache[key])
    if res.empty:
        return pd.DataFrame(columns=cols), warn
    if "query" not in res:
        res["query"] = next(iter(queries))
    res = res[[c for c in cols if c in res.columns]].sort_values(["query", "p_value"]).reset_index(drop=True)
    return res, warn
