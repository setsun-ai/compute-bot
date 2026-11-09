"""Entry point: `python -m compute_bot.recipes.omics` in the job's work directory (see docs/RECIPE_CONTRACT.md)."""
import os
import sys

_threads = os.environ.get("COMPUTE_THREADS") or "1"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _threads)
os.environ.setdefault("MPLBACKEND", "Agg")

import json  # noqa: E402
from pathlib import Path  # noqa: E402

from compute_bot.recipes import RECIPES  # noqa: E402
from compute_bot.recipes.omics import parse, pipeline  # noqa: E402

SUMMARY_MAX = 2500


def load_params(work):
    params = {k: p.default for k, p in RECIPES["omics"].params.items()}
    path = work / "params.json"
    if path.exists():
        params.update(json.loads(path.read_text()))
    return params


def run(work):
    params = load_params(work)
    threads = max(1, int(_threads))
    pipeline.progress("wczytywanie danych")
    datasets = parse.load_datasets(work, group_col=params["group"], layer=params["layer"])
    for ds in datasets:
        pipeline.validate(ds, params)
    root = work / "wyniki"
    root.mkdir(exist_ok=True)
    multi = len(datasets) > 1
    results = []
    for ds in datasets:
        out = root / ds.layer if multi else root
        results.append(pipeline.run_layer(ds, params, out, threads=threads, cache_dir=root / "cache"))
    (root / "kroki.txt").write_text("\n".join(r["log"] for r in results))
    budget = (SUMMARY_MAX - 80) // len(results)
    parts = [pipeline.layer_summary(ds, r, params, budget=budget) for ds, r in zip(datasets, results, strict=True)]
    summary = "\n\n".join(parts) + "\nSzczegóły: wyniki/kroki.txt, tabele CSV i wykresy w wyniki/."
    (work / "summary.txt").write_text(summary[:SUMMARY_MAX])
    per = 4 if not multi else 2
    hl = []
    for r in results:
        hl += [p for p in r["highlights"] if p.exists()][:per]
    (work / "highlights.txt").write_text("\n".join(str(p.relative_to(work)) for p in hl[:4]) + "\n")
    pipeline.progress("gotowe")


def main():
    try:
        run(Path.cwd())
    except parse.InputError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
