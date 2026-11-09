"""Tests for the `md` recipe: pure functions on known inputs, build/stage idempotency with a fake `gmx`, and a slow
real end-to-end run (1L2Y from RCSB) that is interrupted mid-production and resumed."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from compute_bot.recipes import RECIPES
from compute_bot.recipes.md import gmxtools as G
from compute_bot.recipes.md import mdp as M

REPO = Path(__file__).resolve().parents[2]
BIN = REPO / "compute_bot" / "bin"


def params(**given):
    return RECIPES["md"].validate({k: str(v) for k, v in given.items()})


# --- mdp generation -------------------------------------------------------------------------------------------------

def test_nsteps_from_ps():
    assert M.nsteps(50) == 25000
    assert M.nsteps(0.004 * 1000) == 2000
    assert M.nsteps(1000) == 500000


def test_default_plan_has_four_stages_with_continuity():
    plan = M.plan_stages(params())
    assert [s.name for s in plan] == ["em", "nvt", "npt", "md"]
    em, nvt, npt, md = plan
    assert (nvt.nsteps, npt.nsteps, md.nsteps) == (25000, 25000, 500000)
    assert [s.gen_vel for s in plan] == [False, True, False, False]
    assert [s.posres for s in plan] == [False, True, True, False]
    assert em.grompp == ["-f", "em.mdp", "-c", "ions.gro", "-p", "topol.top"]
    assert nvt.grompp == ["-f", "nvt.mdp", "-c", "em.gro", "-r", "em.gro", "-p", "topol.top"]
    assert npt.grompp == ["-f", "npt.mdp", "-c", "nvt.gro", "-r", "nvt.gro", "-t", "nvt.cpt", "-p", "topol.top"]
    assert md.grompp == ["-f", "md.mdp", "-c", "npt.gro", "-t", "npt.cpt", "-p", "topol.top"]


def test_mdp_settings_per_stage():
    p = params(T=600, seed=1234)
    mdps = {s.name: M.parse_mdp(M.make_mdp(s, p["ff"], p["seed"])) for s in M.plan_stages(p)}
    em, nvt, npt, md = mdps["em"], mdps["nvt"], mdps["npt"], mdps["md"]
    assert em["integrator"] == "steep" and float(em["emtol"]) == 1000
    assert nvt["define"] == "-DPOSRES" and npt["define"] == "-DPOSRES" and "define" not in md
    assert nvt["gen_vel"] == "yes" and nvt["gen_seed"] == "1234" and nvt["continuation"] == "no"
    assert npt["gen_vel"] == "no" and md["gen_vel"] == "no"
    assert npt["continuation"] == "yes" and md["continuation"] == "yes"
    for m in (nvt, npt, md):
        assert m["tc-grps"] == "Protein Non-Protein"
        assert m["dt"] == "0.002" and m["constraints"] == "h-bonds" and m["coulombtype"] == "PME"
        assert m["tcoupl"] == "V-rescale"
    assert nvt["pcoupl"] == "no"
    assert npt["pcoupl"] == "C-rescale" and npt["refcoord_scaling"] == "com"
    assert md["pcoupl"] == "C-rescale"
    # equilibrate at room temperature, then heat to T in production
    assert nvt["ref_t"] == "300 300" and npt["ref_t"] == "300 300" and md["ref_t"] == "600 600"
    assert nvt["gen_temp"] == "300"
    assert md["nsteps"] == "500000"
    # every energy frame must be a multiple of nstcalcenergy
    for m in (nvt, npt, md):
        assert int(m["nstenergy"]) % int(m["nstcalcenergy"]) == 0


def test_low_temperature_equilibrates_at_target():
    p = params(T=280)
    nvt = M.parse_mdp(M.make_mdp(M.plan_stages(p)[1], p["ff"], p["seed"]))
    assert nvt["ref_t"] == "280 280"


def test_velocities_move_to_first_dynamic_stage_when_nvt_skipped():
    plan = M.plan_stages(params(nvt_ps=0))
    assert [s.name for s in plan] == ["em", "npt", "md"]
    npt = plan[1]
    assert npt.gen_vel and "-t" not in npt.grompp
    assert npt.grompp[:6] == ["-f", "npt.mdp", "-c", "em.gro", "-r", "em.gro"]
    m = M.parse_mdp(M.make_mdp(npt, "amber99sb-ildn", -1))
    assert m["gen_vel"] == "yes" and m["continuation"] == "no"
    only_md = M.plan_stages(params(nvt_ps=0, npt_ps=0))
    assert [s.name for s in only_md] == ["em", "md"] and only_md[1].gen_vel


def test_water_auto_pairing():
    assert M.resolve_water("amber99sb-ildn", "auto") == "tip3p"
    assert M.resolve_water("amber03", "auto") == "tip3p"
    assert M.resolve_water("charmm27", "auto") == "tip3p"
    assert M.resolve_water("oplsaa", "auto") == "tip4p"
    assert M.resolve_water("gromos54a7", "auto") == "spc"
    assert M.resolve_water("amber99sb-ildn", "spce") == "spce"
    assert M.solvent_box("tip4p") == "tip4p.gro" and M.solvent_box("tip3p") == "spc216.gro"


def test_force_field_specific_cutoffs():
    s = M.plan_stages(params(ff="charmm27"))[1]
    charmm = M.parse_mdp(M.make_mdp(s, "charmm27", -1))
    assert charmm["vdw-modifier"] == "force-switch" and charmm["rvdw"] == "1.2" and charmm["DispCorr"] == "no"
    amber = M.parse_mdp(M.make_mdp(s, "amber99sb-ildn", -1))
    assert amber["rvdw"] == "1.0" and amber["DispCorr"] == "EnerPres" and "vdw-modifier" not in amber


def test_nvt_production_ensemble_option():
    p = dict(params(), ensemble="nvt")  # forward-compatible key, not in the registry yet
    md = M.parse_mdp(M.make_mdp(M.plan_stages(p)[-1], p["ff"], -1))
    assert md["pcoupl"] == "no"


# --- parsers --------------------------------------------------------------------------------------------------------

XVG = """# GROMACS energy
@    title "GROMACS Energies"
@    xaxis  label "Time (ps)"
@    yaxis  label "(K), (bar)"
@TYPE xy
@ s0 legend "Temperature"
@ s1 legend "Pressure"
    0.000000  302.39   -12.5
    1.000000  299.23    4.0
&
    2.000000  300.00    1.5
"""


def test_parse_xvg():
    xvg = G.parse_xvg(XVG)
    assert xvg.legends == ["Temperature", "Pressure"]
    assert xvg.xlabel == "Time (ps)"
    assert xvg.data.shape == (3, 3)
    assert xvg.column("Pressure").tolist() == [-12.5, 4.0, 1.5]
    assert xvg.x.tolist() == [0.0, 1.0, 2.0]


LOG_TAIL = """
               Core t (s)   Wall t (s)        (%)
       Time:    11191.594     1119.160     1000.0
                 (ns/day)    (hour/ns)    (ms/step)  (Matom*steps/s)
Performance:       38.601        0.622        4.477            2.861
Finished mdrun on rank 0 Thu Nov  5 15:49:01 2026
"""


def test_parse_performance():
    assert G.parse_performance(LOG_TAIL) == pytest.approx(38.601)
    assert G.parse_performance("Performance:  1.0  2.0\n...\nPerformance:  5.5  4.3\n") == pytest.approx(5.5)
    assert G.parse_performance("no performance here") is None


def test_parse_em_log():
    ok = G.parse_em_log("Steepest Descents converged to Fmax < 1000 in 345 steps\n"
                        "Potential Energy  = -8.8918641e+04\nMaximum force     =  9.5794910e+02 on atom 302\n")
    assert ok == {"converged": True, "steps": 345, "epot": pytest.approx(-88918.641), "fmax": pytest.approx(957.9491),
                  "emtol": 1000.0}
    bad = G.parse_em_log("Steepest Descents converged to machine precision in 1234 steps,\n"
                         "but did not reach the requested Fmax < 1000.\n"
                         "Potential Energy  = -1.0e+04\nMaximum force     =  2.5e+03 on atom 7\n")
    assert bad["converged"] is False and bad["steps"] == 1234 and bad["fmax"] == pytest.approx(2500)


def test_parse_mdrun_progress():
    assert G.parse_progress("step 5000, will finish Thu Nov  5 15:49:01 2026") == {
        "step": 5000, "finish": "Thu Nov  5 15:49:01 2026"}
    assert G.parse_progress("imb F  3% step 100, remaining wall clock time:    14 s") == {"step": 100, "remaining": 14}
    em = G.parse_progress("Step=   12, Dmax= 1.2e-02 nm, Epot= -6.83e+04 Fmax= 1.64e+04, atom= 1234")
    assert em == {"step": 12, "epot": pytest.approx(-68300.0), "fmax": pytest.approx(16400.0)}
    assert G.parse_progress("Reading file nvt.tpr") is None


def test_parse_topology_and_box():
    top = ("[ system ]\nProtein in water\n\n[ molecules ]\n; Compound #mols\n"
           "Protein_chain_A     1\nSOL  1866\nNA 5\nCL 6\n")
    assert G.parse_molecules(top) == [("Protein_chain_A", 1), ("SOL", 1866), ("NA", 5), ("CL", 6)]
    gro = ("t\n    2\n    1ALA     CA    1   0.000   0.000   0.000\n    1ALA     CB    2   0.100   0.000   0.000\n"
           "   4.39229   4.39229   3.10582   0.00000   0.00000   0.00000   0.00000   2.19615   2.19615\n")
    box = G.parse_gro(gro)
    assert box["atoms"] == 2
    assert box["lengths"] == pytest.approx([4.39229, 4.39229, 4.39229], abs=1e-3)
    assert box["volume"] == pytest.approx(4.39229 * 4.39229 * 3.10582)


def test_grompp_charge():
    assert G.parse_net_charge("  System has non-zero total charge: 1.000000\n") == pytest.approx(1.0)
    assert G.parse_net_charge("nothing to see") == 0.0


# --- structure check ------------------------------------------------------------------------------------------------

def _atom(serial, name, resn, resi, x, y, z, elem, het=False, altloc=" ", occ=1.0, chain="A"):
    rec = "HETATM" if het else "ATOM  "
    return (f"{rec}{serial:5d} {name:<4}{altloc}{resn:>3} {chain}{resi:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}"
            f"  0.00          {elem:>2}")


def synthetic_pdb():
    """Ala-Gly-Ser (Ser misses OG), Ala CB in two altlocs, SO4 + 2 waters, two NMR models (model 2 shifted 10 A)."""
    backbone = {"N": (0.0, 1.0, 0.0), "CA": (1.0, 0.0, 0.0), "C": (2.3, 0.6, 0.0), "O": (2.5, 1.8, 0.0)}
    residues = [("ALA", {"CB": (1.0, -0.8, 1.2)}), ("GLY", {}), ("SER", {"CB": (1.0, -0.8, 1.2)})]
    lines = ["SEQRES   1 A    3  ALA GLY SER"]
    for model, shift in ((1, 0.0), (2, 10.0)):
        lines.append(f"MODEL     {model:4d}")
        serial = 1
        for i, (resn, side) in enumerate(residues, start=1):
            dx = (i - 1) * 3.8 + shift
            for name, (x, y, z) in backbone.items():
                lines.append(_atom(serial, name, resn, i, x + dx, y, z, name[0]))
                serial += 1
            for name, (x, y, z) in side.items():
                if resn == "ALA":
                    lines.append(_atom(serial, name, resn, i, x + dx, y, z, "C", altloc="A", occ=0.6))
                    lines.append(_atom(serial + 1, name, resn, i, x + dx + 0.3, y, z, "C", altloc="B", occ=0.4))
                    serial += 2
                else:
                    lines.append(_atom(serial, name, resn, i, x + dx, y, z, "C"))
                    serial += 1
        lines.append("TER")
        lines.append(_atom(serial, "S", "SO4", 101, 20.0 + shift, 5.0, 5.0, "S", het=True))
        lines.append(_atom(serial + 1, "O", "HOH", 201, 25.0 + shift, 5.0, 5.0, "O", het=True))
        lines.append(_atom(serial + 2, "O", "HOH", 202, 28.0 + shift, 5.0, 5.0, "O", het=True))
        lines.append("ENDMDL")
    lines.append("END")
    return "\n".join(lines) + "\n"


def test_inspect_structure(tmp_path):
    pytest.importorskip("gemmi")
    from compute_bot.recipes.md import structure as S
    path = tmp_path / "toy.pdb"
    path.write_text(synthetic_pdb())
    info = S.inspect_structure(path)
    assert info["models"] == 2
    assert [c["name"] for c in info["chains"]] == ["A"]
    assert info["chains"][0]["residues"] == 3 and info["chains"][0]["sequence"] == "AGS"
    assert info["waters"] == 2
    assert info["hetero"] == {"SO4": 1}
    assert info["altloc_atoms"] == 2
    assert info["missing_atoms"] == {"A:3 SER": ["OG"]}
    assert info["hydrogens"] is False
    text = S.describe_structure(info, model=2)
    assert "2 modele" in text and "SO4" in text and "OG" in text


def test_prepare_structure_selects_model_and_strips(tmp_path):
    pytest.importorskip("gemmi")
    pytest.importorskip("pdbfixer")
    from compute_bot.recipes.md import structure as S
    path = tmp_path / "toy.pdb"
    path.write_text(synthetic_pdb())
    out = tmp_path / "clean.pdb"
    notes = S.prepare_structure(path, out, model=2, chains="")
    text = out.read_text()
    atoms = [ln for ln in text.splitlines() if ln.startswith(("ATOM", "HETATM"))]
    assert not any(" HOH " in ln or " SO4 " in ln for ln in atoms)
    names = [(ln[17:20], ln[12:16].strip()) for ln in atoms]
    assert ("SER", "OG") in names  # repaired by pdbfixer
    assert sum(1 for r, n in names if (r, n) == ("ALA", "CB")) == 1  # one altloc kept
    assert all(float(ln[30:38]) > 9.0 for ln in atoms if ln[12:16].strip() == "N" and ln[17:20] == "ALA")  # model 2
    joined = " ".join(notes)
    assert "wod" in joined.lower()
    assert "SO4" in joined and "pdbfixer" in joined.lower()


def test_prepare_structure_bad_choices(tmp_path):
    pytest.importorskip("gemmi")
    from compute_bot.recipes.md import structure as S
    path = tmp_path / "toy.pdb"
    path.write_text(synthetic_pdb())
    with pytest.raises(S.RecipeError, match="model"):
        S.prepare_structure(path, tmp_path / "c.pdb", model=5, chains="")
    with pytest.raises(S.RecipeError, match="łańcuch"):
        S.prepare_structure(path, tmp_path / "c.pdb", model=1, chains="B")


# --- idempotency with a fake gmx ------------------------------------------------------------------------------------

FAKE_GMX = r'''#!PYTHON
"""Pretend gmx: records the call and writes plausible outputs."""
import os, sys
args = sys.argv[1:]
with open(os.environ["FAKE_GMX_LOG"], "a") as fh:
    fh.write(" ".join(args) + "\n")
tool, rest = args[0], args[1:]
def opt(name, default=None):
    return rest[rest.index(name) + 1] if name in rest else default
GRO = "fake\n    3\n    1ALA     CA    1   0.000   0.000   0.000\n    2SOL     OW    2   1.000   1.000   1.000\n" \
      "    3NA      NA    3   2.000   2.000   2.000\n   3.00000   3.00000   3.00000\n"
if tool == "pdb2gmx":
    open(opt("-o"), "w").write(GRO)
    open(opt("-p"), "w").write("[ system ]\nfake\n\n[ molecules ]\nProtein_chain_A 1\n")
    open(opt("-i", "posre.itp"), "w").write("; posre\n")
elif tool in ("editconf", "solvate"):
    open(opt("-o"), "w").write(GRO)
    if tool == "solvate":
        open(opt("-p"), "a").write("SOL 100\n")
elif tool == "grompp":
    open(opt("-o"), "w").write("tpr")
    sys.stderr.write("System has non-zero total charge: 1.000000\n")
elif tool == "genion":
    sys.stdin.read()
    open(opt("-o"), "w").write(GRO)
    top = open(opt("-p")).read().replace("SOL 100", "SOL 95\nNA 2\nCL 3")
    open(opt("-p"), "w").write(top)
elif tool == "mdrun":
    name = opt("-deffnm")
    if "-cpi" not in rest:
        open(name + ".cpt", "w").write("cpt")
    open(name + ".gro", "w").write(GRO)
    open(name + ".log", "w").write("Performance:   7.5   3.2\nFinished mdrun on rank 0\n")
sys.exit(0)
'''


@pytest.fixture
def fake_gmx(tmp_path, monkeypatch):
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    script = fakebin / "gmx"
    script.write_text(FAKE_GMX.replace("#!PYTHON", "#!" + sys.executable))
    script.chmod(0o755)
    log = tmp_path / "gmx_calls.log"
    log.write_text("")
    monkeypatch.setenv("FAKE_GMX_LOG", str(log))
    monkeypatch.setenv("PATH", f"{fakebin}{os.pathsep}{BIN}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("COMPUTE_THREADS", "2")
    return log


def _calls(log):
    return [ln for ln in log.read_text().splitlines() if ln]


def test_build_is_idempotent(tmp_path, fake_gmx):
    from compute_bot.recipes.md import pipeline as P
    gdir = tmp_path / "gromacs"
    gdir.mkdir()
    (gdir / "clean.pdb").write_text("ATOM\n")
    p = params(salt=0.1, seed=7)
    env = G.gmx_env(2)
    system = P.build_system(gdir, p, env)
    first = _calls(fake_gmx)
    assert [c.split()[0] for c in first] == ["pdb2gmx", "editconf", "solvate", "grompp", "genion"]
    assert "-ignh" in first[0] and "-water tip3p" in first[0] and "-ff amber99sb-ildn" in first[0]
    assert "-bt dodecahedron" in first[1] and "-d 1 " in first[1]
    assert "-conc 0.1" in first[4] and "-neutral" in first[4] and "-seed 7" in first[4]
    assert system["waters"] == 95 and system["na"] == 2 and system["cl"] == 3
    assert system["charge_protein"] == pytest.approx(1.0) and system["charge_total"] == pytest.approx(0.0)
    assert (gdir / "topol.top").exists() and (gdir / "ions.gro").exists()
    again = P.build_system(gdir, p, env)
    assert _calls(fake_gmx) == first  # nothing re-run
    assert again == system


def test_stages_skip_finished_and_resume_from_checkpoint(tmp_path, fake_gmx, capsys):
    from compute_bot.recipes.md import pipeline as P
    gdir = tmp_path / "gromacs"
    gdir.mkdir()
    p = params(nvt_ps=2, npt_ps=2, ns=0.004)
    plan = M.plan_stages(p)
    env = G.gmx_env(2)
    P.write_mdps(gdir, plan, p)
    P.run_stages(gdir, plan, env)
    first = _calls(fake_gmx)
    assert [c.split()[0] for c in first] == ["grompp", "mdrun"] * 4
    assert all("-nt 2" in c for c in first if c.startswith("mdrun"))
    assert "PROGRESS:" in capsys.readouterr().out
    P.run_stages(gdir, plan, env)
    assert _calls(fake_gmx) == first  # finished stages are skipped
    # production interrupted: no final .gro, checkpoint and tpr remain
    (gdir / "md.gro").unlink()
    P.run_stages(gdir, plan, env)
    new = _calls(fake_gmx)[len(first):]
    assert len(new) == 1 and new[0].startswith("mdrun") and "-cpi md.cpt" in new[0] and "-deffnm md" in new[0]


def test_missing_input_is_a_readable_error(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "params.json").write_text(json.dumps(params()))
    env = dict(os.environ, PYTHONPATH=str(REPO), PATH=f"{BIN}{os.pathsep}{os.environ['PATH']}")
    proc = subprocess.run([sys.executable, "-m", "compute_bot.recipes.md"], cwd=work, env=env, capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode != 0
    assert proc.stderr.strip().splitlines()[-1].startswith("ERROR: ")


# --- slow: real GROMACS run on trp-cage -----------------------------------------------------------------------------

def _recipe_env():
    gmx = shutil.which("gmx")
    env_bin = str(Path(gmx).parent) if gmx else str(Path(sys.executable).parent)
    return dict(os.environ, PYTHONPATH=str(REPO), COMPUTE_THREADS="4",
                PATH=os.pathsep.join([env_bin, str(BIN), os.environ.get("PATH", "")]))


@pytest.mark.slow
def test_end_to_end_1l2y_with_interrupt_and_resume(tmp_path):
    pytest.importorskip("gemmi")
    pytest.importorskip("matplotlib")
    env = _recipe_env()
    if not shutil.which("gmx", path=env["PATH"]):
        pytest.skip("gmx not on PATH")
    work = tmp_path / "work"
    work.mkdir()
    p = RECIPES["md"].validate({"pdb": "1L2Y", "nvt_ps": "2", "npt_ps": "2", "ns": "0.03", "seed": "11"})
    (work / "params.json").write_text(json.dumps(p))
    cmd = [sys.executable, "-m", "compute_bot.recipes.md"]
    gdir = work / "gromacs"

    # run 1: interrupted in the middle of production (TERM to the whole group, like a shutdown)
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
    seen = []
    for line in proc.stdout:
        seen.append(line)
        if line.startswith("PROGRESS:") and "produkcja" in line and "krok" in line:
            break
    else:
        proc.wait()
        pytest.fail("production never started:\n" + "".join(seen[-40:]))
    os.killpg(proc.pid, signal.SIGTERM)
    proc.wait(timeout=60)
    for _ in range(120):  # wait for mdrun to write its checkpoint and exit
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.5)
    assert (gdir / "md.cpt").exists(), "".join(seen[-40:])
    assert not (gdir / "md.gro").exists()
    t_first = time.time() - t0

    # run 2: resumes production from the checkpoint and analyses
    t1 = time.time()
    run2 = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=1800)
    t_second = time.time() - t1
    assert run2.returncode == 0, run2.stdout[-3000:] + run2.stderr[-3000:]
    assert "wznawiam md z checkpointu" in run2.stdout
    assert "em już policzony" in run2.stdout and "npt już policzony" in run2.stdout
    log = (gdir / "md.log").read_text()
    assert "Finished mdrun" in log
    summary = (work / "summary.txt").read_text()
    assert len(summary) <= 2500
    for word in ("1L2Y", "amber99sb-ildn", "tip3p", "Minimalizacja", "RMSD", "ns/dzień", "Na", "Cl", "wód"):
        assert word in summary, word
    highlights = [ln for ln in (work / "highlights.txt").read_text().splitlines() if ln]
    assert 1 <= len(highlights) <= 4
    for rel in highlights:
        assert (work / rel).stat().st_size > 5000
    results = work / "wyniki"
    for name in ("energia_nvt.csv", "energia_npt.csv", "energia_md.csv", "rmsd.csv", "rmsf.csv", "rg.csv",
                 "wiazania_h.csv", "uklad.csv", "struktura.txt", "bialko.xtc", "bialko_klatki.pdb"):
        assert (results / name).exists(), name
    assert (results / "bialko_klatki.pdb").stat().st_size < 20e6
    system = json.loads((gdir / "system.json").read_text())
    assert system["charge_total"] == pytest.approx(0.0, abs=1e-3)
    assert system["na"] > 0 and system["cl"] > 0

    # run 3: everything is already done
    t2 = time.time()
    run3 = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=600)
    t_third = time.time() - t2
    assert run3.returncode == 0, run3.stderr[-3000:]
    for stage in ("em", "nvt", "npt", "md"):
        assert f"{stage} już policzony" in run3.stdout
    assert "wznawiam" not in run3.stdout
    build_log = (gdir / "build.log").read_text()
    assert build_log.count("$ gmx pdb2gmx") == 1 and build_log.count("$ gmx genion") == 1
    print(textwrap.dedent(f"""
        timing: interrupted run {t_first:.0f} s, resumed run + analysis {t_second:.0f} s, no-op rerun {t_third:.0f} s
    """))
