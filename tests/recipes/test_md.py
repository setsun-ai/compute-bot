"""Tests for the `md` recipe: pure functions on known inputs, build/stage idempotency with a fake `gmx`, and a slow
real end-to-end run (1L2Y from RCSB) that is interrupted mid-production and resumed."""
from __future__ import annotations

import json
import os
import re
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


def test_fatal_error_names_the_grompp_warning():
    text = ("Generating 1-4 interactions: fudge = 0.5\nWARNING 1 [file md.mdp]:\n"
            "  For proper integration of the Nose-Hoover thermostat, tau-t (0.1) should\n"
            "  be at least 20 times larger than nsttcouple*dt (0.1)\nNumber of degrees of freedom\n"
            "-------------------------------------------------------\nProgram:     gmx grompp\n\n"
            "Fatal error:\nToo many warnings (1).\nIf you are sure all warnings are harmless, use the -maxwarn "
            "option.\n\nFor more information and tips for troubleshooting\n")
    msg = G.fatal_error(text)
    assert "Nose-Hoover" in msg and "Too many warnings" in msg


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
elif tool == "hbond":
    open(opt("-o", "hbond.ndx"), "w").write("ndx")  # real gmx hbond writes this even when not asked
    open(opt("-num"), "w").write("xvg")
elif tool == "grompp":
    open(opt("-po", "mdout.mdp"), "w").write("mdout")
    open(opt("-o"), "w").write("tpr")
    sys.stderr.write("System has non-zero total charge: 1.000000\n")
elif tool == "genion":
    sys.stdin.read()
    open(opt("-o"), "w").write(GRO)
    top = open(opt("-p")).read().replace("SOL 100", "SOL 95\nNA 2\nCL 3")
    open(opt("-p"), "w").write(top)
elif tool == "convert-tpr":
    open(opt("-o"), "w").write("tpr-extended")
elif tool == "mdrun" and "-noappend" in rest:
    import glob
    name = opt("-deffnm")
    part = f"{name}.part{2 + len(glob.glob(name + '.part*.log')):04d}"
    for ext in (".xtc", ".edr"):
        open(part + ext, "w").write("x")
    open(part + ".gro", "w").write(GRO)
    open(part + ".log", "w").write("Performance:   7.5   3.2\nFinished mdrun on rank 0\n")
    open(name + ".cpt", "w").write("cpt")
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


# --- mdp overrides --------------------------------------------------------------------------------------------------

def test_parse_mdp_overrides():
    assert M.parse_overrides("") == []
    assert M.parse_overrides("nstxout-compressed=500; tcoupl = nose-hoover;") == [
        ("nstxout-compressed", "500"), ("tcoupl", "nose-hoover")]
    assert M.parse_overrides("tc-grps=Protein Non-Protein") == [("tc-grps", "Protein Non-Protein")]
    for bad in ("foo", "=1", "a b=1", "x=", "nsteps=1=2", "1abc=3"):
        with pytest.raises(G.RecipeError, match="mdp"):
            M.parse_overrides(bad)


def test_overrides_change_only_production():
    p = params(mdp="nstxout_compressed=500;tcoupl=nose-hoover;nstcalcenergy=50;cos-acceleration=0.1")
    plan = M.plan_stages(p)
    over = M.parse_overrides(p["mdp"])
    texts = {s.name: M.make_mdp(s, p["ff"], p["seed"], over if s.name == "md" else None) for s in plan}
    md = M.parse_mdp(texts["md"])
    assert md["tcoupl"] == "nose-hoover" and md["nstxout-compressed"] == "500" and md["nstcalcenergy"] == "50"
    assert md["cos-acceleration"] == "0.1"
    assert "nstxout_compressed" not in md  # replaced, not duplicated
    assert texts["md"].count("nstxout") == 1
    assert M.parse_mdp(texts["npt"])["tcoupl"] == "V-rescale"


def test_check_params_engine_rules():
    from compute_bot.recipes.md import pipeline as P
    assert P.check_params(params()) == []
    with pytest.raises(G.RecipeError, match="mdp"):
        P.check_params(params(engine="openmm", mdp="tcoupl=nose-hoover"))
    with pytest.raises(G.RecipeError, match="OpenMM"):
        P.check_params(params(engine="openmm", ff="gromos54a7"))
    with pytest.raises(G.RecipeError, match="OpenMM"):
        P.check_params(params(engine="openmm", ff="oplsaa"))
    notes = P.check_params(params(engine="openmm", ff="charmm27"))
    assert any("CHARMM36" in n for n in notes)
    notes = P.check_params(params(extend_ns=1, mdp="tcoupl=nose-hoover"))
    assert any("mdp" in n for n in notes)


def test_openmm_forcefield_mapping():
    from compute_bot.recipes.md import omm as O
    files, model, _ = O.openmm_forcefield("amber99sb-ildn", "auto")
    assert files == ["amber99sbildn.xml", "tip3p.xml"] and model == "tip3p"
    files, model, notes = O.openmm_forcefield("charmm27", "auto")
    assert files == ["charmm36.xml", "charmm36/water.xml"] and model == "tip3p" and "CHARMM36" in notes[0]
    files, model, notes = O.openmm_forcefield("amber03", "tip4p")
    assert files == ["amber03.xml", "tip4pew.xml"] and model == "tip4pew" and "TIP4P-Ew" in notes[0]
    with pytest.raises(G.RecipeError, match="spc"):
        O.openmm_forcefield("amber99sb", "spc")


# --- OpenMM bookkeeping ---------------------------------------------------------------------------------------------

def test_openmm_frames_kept_after_restart():
    from compute_bot.recipes.md import omm as O
    # part 1 started at 0 and got to 1500 before a crash, last checkpoint at 1200; part 2 resumed at 1200
    kept = O.kept_frames([0, 1200], [15, 10], nst=100, end=2000)
    assert kept[0] == list(range(100, 1201, 100))
    assert kept[1] == list(range(1300, 2001, 100))
    # an extension starts mid-interval: frames are absolute multiples of nst
    assert O.kept_frames([2050], [3], nst=100, end=2400) == [[2100, 2200, 2300]]


def test_openmm_box_matches_editconf():
    import numpy as np

    from compute_bot.recipes.md import omm as O
    for shape, volume in (("cubic", 1.0), ("dodecahedron", 0.5 * 2 ** 0.5), ("octahedron", 4 / 9 * 3 ** 0.5)):
        v = np.array(O.box_vectors(shape, 4.0))
        assert np.linalg.norm(v, axis=1) == pytest.approx([4.0, 4.0, 4.0])
        assert abs(np.linalg.det(v)) == pytest.approx(volume * 64)
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 0.5]] + [[0.1, 0.1, 0.1]] * 60)
    assert O.diameter(pts) == pytest.approx(5 ** 0.5)


def test_parse_state_data_csv():
    from compute_bot.recipes.md import omm as O
    text = ('#"Step","Time (ps)","Potential Energy (kJ/mole)","Temperature (K)","Box Volume (nm^3)",'
            '"Density (g/mL)"\n100,0.2,-80000.5,299.1,60.1,1.012\n200,0.4,-80010.0,301.0,60.0,1.013\n')
    cols = O.parse_state_csv(text)
    assert cols["Step"].tolist() == [100, 200]
    assert cols["Potential"].tolist() == [-80000.5, -80010.0]
    assert cols["Density"].tolist() == pytest.approx([1012.0, 1013.0])  # kg/m3 like gmx energy
    assert cols["Volume"].tolist() == [60.1, 60.0] and "Temperature" in cols


def test_progress_counts_from_first_step():
    stage = M.Stage("md_ext1", "przedłużenie 1", 1000, 300.0, first_step=2000)
    prog = G.Progress([stage], every=0, now=iter([0.0, 10.0]).__next__)
    prog.start(stage)
    assert prog.update({"step": 2000}) .startswith("PROGRESS: etap 1/1: przedłużenie 1 0%")
    assert "50%" in prog.update({"step": 2500})


# --- extension of a finished job ------------------------------------------------------------------------------------

def test_claim_extension_segment(tmp_path):
    from compute_bot.recipes.md import extend as X
    d = tmp_path / "gromacs"
    d.mkdir()
    assert X.claim_segment(d, tmp_path / "w1") == ("md_ext1", 1)
    assert X.claim_segment(d, tmp_path / "w1") == ("md_ext1", 1)  # same job: same segment
    (d / "md_ext1.done.json").write_text("{}")
    (d / "md_ext2.part0002.log").write_text("left over by an unfinished parent")
    assert X.claim_segment(d, tmp_path / "w2") == ("md_ext2", 2)  # a child job: next segment
    assert not (d / "md_ext2.part0002.log").exists()
    assert X.production_segments(d) == ["md", "md_ext1"]


def _finished_gromacs_job(work):
    gdir = work / "gromacs"
    gdir.mkdir(parents=True)
    for name in ("md.tpr", "md.cpt", "md.xtc", "md.edr", "md.gro", "topol.top"):
        (gdir / name).write_text(name)
    (gdir / "md.log").write_text("Performance:   7.5   3.2\nFinished mdrun on rank 0\n")
    (gdir / "md.mdp").write_text("nsteps = 2000\nref_t = 300 300\npcoupl = C-rescale\n")
    (gdir / "analysis.log").write_text("old log\n")
    (work / "summary.txt").write_text("parent summary\n")
    return gdir


def _link_tree(src: Path, dst: Path):
    """What the bot does for an extension job: every file of work/ hard-linked into the new work dir."""
    for path in src.rglob("*"):
        target = dst / path.relative_to(src)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)


def _snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_gromacs_extension_call_sequence_and_parent_untouched(tmp_path, fake_gmx):
    from compute_bot.recipes.md import extend as X
    parent = tmp_path / "parent"
    _finished_gromacs_job(parent)
    before = _snapshot(parent)
    child = tmp_path / "child"
    _link_tree(parent, child)
    gdir = child / "gromacs"
    env = G.gmx_env(2)
    p = params(extend_ns=1)
    seg = X.extend_gromacs(child, gdir, p, env)
    assert seg == "md_ext1"
    calls = _calls(fake_gmx)
    assert calls[0].startswith("convert-tpr -s md.tpr -extend 1000 -o ")
    assert calls[1].startswith("mdrun") and "-deffnm md_ext1" in calls[1] and "-cpi md.cpt" in calls[1]
    assert "-noappend" in calls[1] and "-nt 2" in calls[1] and "-s md_ext1.tpr" in calls[1]
    assert (gdir / "md_ext1.done.json").exists()
    assert X.production_segments(gdir) == ["md", "md_ext1"]
    # rerun: nothing to do
    X.extend_gromacs(child, gdir, p, env)
    assert _calls(fake_gmx) == calls
    # interrupted extension: resumes from its own checkpoint, still without appending
    (gdir / "md_ext1.done.json").unlink()
    (gdir / "md_ext1.part0002.log").write_text("step 10\n")
    X.extend_gromacs(child, gdir, p, env)
    last = _calls(fake_gmx)[-1]
    assert last.startswith("mdrun") and "-cpi md_ext1.cpt" in last and "-noappend" in last
    assert len(_calls(fake_gmx)) == len(calls) + 1  # the tpr is not rebuilt
    # a grandchild extends the extension
    grand = tmp_path / "grand"
    _link_tree(child, grand)
    X.extend_gromacs(grand, grand / "gromacs", p, env)
    convert, mdrun = _calls(fake_gmx)[-2:]
    assert convert.startswith("convert-tpr -s md_ext1.tpr -extend 1000")
    assert "-deffnm md_ext2" in mdrun and "-cpi md_ext1.cpt" in mdrun
    assert _snapshot(parent) == before  # the parent job's files never change


def test_extension_needs_finished_production(tmp_path, fake_gmx):
    from compute_bot.recipes.md import extend as X
    gdir = tmp_path / "gromacs"
    gdir.mkdir()
    with pytest.raises(G.RecipeError, match="extend_ns"):
        X.extend_gromacs(tmp_path, gdir, params(extend_ns=1), G.gmx_env(2))


def test_gmx_implicit_outputs_do_not_touch_linked_files(tmp_path, fake_gmx):
    parent, child = tmp_path / "p", tmp_path / "c"
    parent.mkdir()
    child.mkdir()
    for name in ("hbond.ndx", "mdout.mdp", "hb.xvg"):
        (parent / name).write_text("parent")
        os.link(parent / name, child / name)
    G.run_gmx(["hbond", "-s", "md.tpr", "-num", "hb.xvg"], child, G.gmx_env(1))
    G.run_gmx(["grompp", "-f", "x.mdp", "-o", "x.tpr"], child, G.gmx_env(1))
    assert all((parent / n).read_text() == "parent" for n in ("hbond.ndx", "mdout.mdp", "hb.xvg"))


def test_safe_writers_break_hard_links(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("parent")
    b = tmp_path / "b.txt"
    os.link(a, b)
    G.write_text(b, "child")
    assert a.read_text() == "parent" and b.read_text() == "child"
    c = tmp_path / "c.log"
    os.link(a, c)
    G.own(c)
    with open(c, "a") as fh:
        fh.write(" more")
    assert a.read_text() == "parent" and c.read_text() == "parent more"
    d = tmp_path / "d.png"
    os.link(a, d)
    G.fresh_output(d).write_bytes(b"png")
    assert a.read_text() == "parent"


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

    # run 1: a power cut (SIGKILL to the whole group) once production has written a checkpoint. Checkpoints come
    # every ~0.3 s here, and the test waits for the file itself instead of guessing from timing (a TERM sent
    # right at production start found no checkpoint on a slower CI runner).
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=work, env=dict(env, MD_CHECKPOINT_MIN="0.005"), stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    seen = []
    while not (gdir / "md.cpt").exists():
        if proc.poll() is not None:
            pytest.fail("production ended before its first checkpoint")
        time.sleep(0.1)
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=60)
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


def _hashes(root: Path) -> dict:
    import hashlib
    return {str(p.relative_to(root)): hashlib.sha1(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _child_job(parent: Path, child: Path, **changes) -> dict:
    _link_tree(parent, child)
    p = dict(json.loads((parent / "params.json").read_text()), **changes)
    (child / "params.json").unlink()  # the bot writes its own params.json, never into the linked one
    (child / "params.json").write_text(json.dumps(p))
    return p


def _production_rows(work: Path) -> int:
    return sum(1 for ln in (work / "wyniki" / "rmsd.csv").read_text().splitlines()[1:] if ln.startswith("md,"))


@pytest.mark.slow
def test_gromacs_extend_1l2y(tmp_path):
    pytest.importorskip("gemmi")
    env = _recipe_env()
    if not shutil.which("gmx", path=env["PATH"]):
        pytest.skip("gmx not on PATH")
    cmd = [sys.executable, "-m", "compute_bot.recipes.md"]
    parent = tmp_path / "parent"
    parent.mkdir()
    p = RECIPES["md"].validate({"pdb": "1L2Y", "nvt_ps": "2", "npt_ps": "2", "ns": "0.004", "seed": "5"})
    (parent / "params.json").write_text(json.dumps(p))
    t0 = time.time()
    base = subprocess.run(cmd, cwd=parent, env=env, capture_output=True, text=True, timeout=1800)
    t_base = time.time() - t0
    assert base.returncode == 0, base.stderr[-3000:]
    rows_before = _production_rows(parent)
    before = _hashes(parent)

    child = tmp_path / "child"
    _child_job(parent, child, extend_ns=0.004)
    t1 = time.time()
    ext = subprocess.run(cmd, cwd=child, env=env, capture_output=True, text=True, timeout=1800)
    t_ext = time.time() - t1
    assert ext.returncode == 0, ext.stdout[-3000:] + ext.stderr[-3000:]
    assert _hashes(parent) == before, "the parent job's files changed"
    gdir = child / "gromacs"
    assert (gdir / "md_ext1.done.json").exists() and list(gdir.glob("md_ext1.part*.xtc"))
    assert _production_rows(child) >= 2 * rows_before - 2
    summary = (child / "summary.txt").read_text()
    assert "Przedłużenie" in summary and "0.008 ns" in summary
    again = subprocess.run(cmd, cwd=child, env=env, capture_output=True, text=True, timeout=600)
    assert again.returncode == 0 and "md_ext1 już policzony" in again.stdout
    assert _hashes(parent) == before
    print(f"\ntiming: base job {t_base:.0f} s, extension + analysis {t_ext:.0f} s")


@pytest.mark.slow
def test_openmm_end_to_end_1l2y_kill_resume_extend(tmp_path):
    pytest.importorskip("openmm")
    pytest.importorskip("pdbfixer")
    pytest.importorskip("MDAnalysis")
    env = dict(_recipe_env(), MD_CHECKPOINT_S="0.3")
    cmd = [sys.executable, "-m", "compute_bot.recipes.md"]
    work = tmp_path / "work"
    work.mkdir()
    p = RECIPES["md"].validate({"pdb": "1L2Y", "engine": "openmm", "nvt_ps": "2", "npt_ps": "2", "ns": "0.01",
                                "seed": "3"})
    (work / "params.json").write_text(json.dumps(p))
    odir = work / "openmm"

    # run 1: killed hard (like a power cut) right after production's first checkpoint; the recipe does it to
    # itself on a test-only switch, so a fast machine can't finish the run before the kill
    t0 = time.time()
    first = subprocess.run(cmd, cwd=work, env=dict(env, MD_TEST_CRASH_AFTER_CHECKPOINT="1"),
                           capture_output=True, text=True, timeout=1800)
    assert first.returncode == -signal.SIGKILL, first.stdout[-3000:] + first.stderr[-3000:]
    assert (odir / "md.chk.xml").exists()
    t_first = time.time() - t0
    assert not (odir / "md.done.json").exists()

    # run 2: resumes from the checkpoint, analyses
    t1 = time.time()
    run2 = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=1800)
    t_second = time.time() - t1
    assert run2.returncode == 0, run2.stdout[-3000:] + run2.stderr[-3000:]
    assert "wznawiam md z checkpointu" in run2.stdout
    assert "em już policzony" in run2.stdout and "npt już policzony" in run2.stdout
    parts = json.loads((odir / "md.parts.json").read_text())
    assert len(parts) == 2 and parts[1]["start"] > 0
    summary = (work / "summary.txt").read_text()
    for word in ("1L2Y", "OpenMM", "amber99sb-ildn", "Minimalizacja", "RMSD", "ns/dzień", "Na", "Cl", "wód"):
        assert word in summary, word
    highlights = [ln for ln in (work / "highlights.txt").read_text().splitlines() if ln]
    assert 1 <= len(highlights) <= 4 and all((work / h).stat().st_size > 5000 for h in highlights)
    results = work / "wyniki"
    for name in ("energia_nvt.csv", "energia_npt.csv", "energia_md.csv", "rmsd.csv", "rmsf.csv", "rg.csv",
                 "wiazania_h.csv", "dssp.csv", "uklad.csv", "struktura.txt", "bialko.xtc", "bialko_klatki.pdb"):
        assert (results / name).exists(), name
    md_rows = _production_rows(work)
    assert md_rows == M.nsteps(10) // M.nice_interval(M.nsteps(10), 1000)  # no duplicated or lost frames
    system = json.loads((odir / "system.json").read_text())
    assert system["charge_total"] == pytest.approx(0.0, abs=1e-3) and system["na"] > 0

    # run 3: nothing left to do
    run3 = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=600)
    assert run3.returncode == 0, run3.stderr[-3000:]
    for stage in ("em", "nvt", "npt", "md"):
        assert f"{stage} już policzony" in run3.stdout

    # extension in a hard-linked child job
    before = _hashes(work)
    child = tmp_path / "child"
    _child_job(work, child, extend_ns=0.004)
    t2 = time.time()
    ext = subprocess.run(cmd, cwd=child, env=env, capture_output=True, text=True, timeout=1800)
    t_ext = time.time() - t2
    assert ext.returncode == 0, ext.stdout[-3000:] + ext.stderr[-3000:]
    assert _hashes(work) == before, "the parent job's files changed"
    assert (child / "openmm" / "md_ext1.done.json").exists()
    assert _production_rows(child) > md_rows
    assert "Przedłużenie" in (child / "summary.txt").read_text()
    print(f"\ntiming: killed run {t_first:.0f} s, resumed run + analysis {t_second:.0f} s, "
          f"extension + analysis {t_ext:.0f} s")


# --- T_eq, stage-prefixed mdp ---------------------------------------------------------------------------------------

def test_equilibration_temperature_option():
    for engine in ("gromacs", "openmm"):
        p = params(T=600, T_eq=320, engine=engine)
        assert {s.name: s.temp for s in M.plan_stages(p)}.items() >= {"nvt": 320, "npt": 320, "md": 600}.items()
    p = params(T=600, T_eq=320, seed=3)
    mdps = {s.name: M.parse_mdp(M.make_mdp(s, p["ff"], p["seed"])) for s in M.plan_stages(p)}
    assert mdps["nvt"]["ref_t"] == "320 320" and mdps["nvt"]["gen_temp"] == "320"
    assert mdps["npt"]["ref_t"] == "320 320" and mdps["md"]["ref_t"] == "600 600"
    # T_eq may also be below T or above 300 when T is low; 0 keeps the old rule
    assert M.plan_stages(params(T=280, T_eq=250))[1].temp == 250
    assert M.plan_stages(params(T=280))[1].temp == 280 and M.plan_stages(params(T=600))[1].temp == 300
    assert M.plan_stages(params(T=600, T_eq=0))[2].temp == 300


def test_stage_prefixed_overrides():
    text = "nstxout-compressed=500;nvt:ref_t=310 310;npt:tau_p=5;em:emtol=500;md:dt=0.001;NPT:tau-p=4"
    over = M.parse_stage_overrides(text)
    assert over == {"md": [("nstxout-compressed", "500"), ("dt", "0.001")], "nvt": [("ref_t", "310 310")],
                    "npt": [("tau_p", "5"), ("tau-p", "4")], "em": [("emtol", "500")]}
    assert M.parse_overrides(text) == over["md"]  # no prefix = production, as before
    assert M.parse_overrides(text, "em") == [("emtol", "500")]
    with pytest.raises(G.RecipeError, match="em, nvt, npt, md") as exc:
        M.parse_stage_overrides("nvt:tau_t=0.2 0.2;prod:dt=0.001")
    assert "prod" in str(exc.value)
    for bad in ("nvt:", "nvt:=1", ":dt=1", "nvt:a b=1"):
        with pytest.raises(G.RecipeError, match="mdp"):
            M.parse_stage_overrides(bad)
    p = params(mdp=text, ns=0.01, nvt_ps=2)
    plan = M.plan_stages(p)
    texts = {s.name: M.parse_mdp(M.make_mdp(s, p["ff"], p["seed"], over.get(s.name))) for s in plan}
    assert float(texts["em"]["emtol"]) == 500
    assert texts["nvt"]["ref_t"] == "310 310" and texts["nvt"]["gen_temp"] == "310"
    assert texts["npt"]["tau_p"] == "4" and "tau-p" not in texts["npt"]
    assert texts["md"]["dt"] == "0.001" and texts["md"]["nstxout-compressed"] == "500"
    nvt, md = plan[1], plan[3]
    assert nvt.temp == 310 and plan[2].temp == 300
    # a new time step keeps the stage's length in ps (unless nsteps is given too)
    assert md.dt == 0.001 and md.nsteps == 10000 and md.ps == pytest.approx(10) and texts["md"]["nsteps"] == "10000"
    md2 = M.plan_stages(params(mdp="md:dt=0.001;nsteps=777"))[-1]
    assert md2.nsteps == 777
    # ref_t in production beats T
    md3 = M.plan_stages(params(T=350, mdp="ref_t=310 310"))[-1]
    assert md3.temp == 310
    with pytest.raises(G.RecipeError, match="liczby"):
        M.plan_stages(params(mdp="md:dt=fast"))


def test_check_params_stage_overrides():
    from compute_bot.recipes.md import pipeline as P
    with pytest.raises(G.RecipeError, match="mdp"):
        P.check_params(params(engine="openmm", mdp="nvt:tau_t=0.2 0.2"))
    with pytest.raises(G.RecipeError, match="dozwolone"):
        P.check_params(params(mdp="eq:tau_t=0.2 0.2"))
    notes = P.check_params(params(extend_ns=1, mdp="npt:tau_p=5"))
    assert any("pominięte" in n for n in notes)
    notes = P.check_params(params(nvt_ps=0, mdp="nvt:tau_t=0.2 0.2;npt:tau_p=5"))
    assert any("nvt_ps=0" in n for n in notes) and not any("npt_ps" in n for n in notes)


def test_summary_lists_overrides_per_stage():
    from compute_bot.recipes.md import analysis as A
    p = params(T=350, mdp="ref_t=310 310;nvt:tau_t=0.2 0.2;em:emtol=500")
    lines = A.override_lines(M.parse_stage_overrides(p["mdp"]), M.plan_stages(p), p)
    text = "\n".join(lines)
    assert "minimalizacja: emtol=500" in text and "NVT: tau_t=0.2 0.2" in text and "produkcja: ref_t=310 310" in text
    assert text.index("minimalizacja") < text.index("NVT") < text.index("produkcja")
    assert "pierwszeństwo przed T=350 K" in text and "310 K" in text
    assert A.override_lines({}, [], p) == []
    assert "pierwszeństwo" not in "\n".join(A.override_lines({"nvt": [("ref_t", "310 310")]}, M.plan_stages(p), p))


def test_actual_plan_reads_temperature_and_time_step_from_the_mdp(tmp_path):
    from compute_bot.recipes.md import pipeline as P
    p = params(T=300, ns=0.004, nvt_ps=2, npt_ps=2)
    plan = M.plan_stages(p)
    for s in plan:
        (tmp_path / f"{s.name}.mdp").write_text(M.make_mdp(s, p["ff"], -1))
    # files of the parent job win over params (here: written with other settings)
    (tmp_path / "nvt.mdp").write_text("nsteps = 1000\ndt = 0.002\nref_t = 315 315\n")
    (tmp_path / "md.mdp").write_text("nsteps = 4000\ndt = 0.001\nref_t = 330 330\n")
    real = {s.name: s for s in P.actual_plan(tmp_path, M.plan_stages(p), "gromacs")}
    assert real["nvt"].temp == 315 and real["nvt"].ps == pytest.approx(2)
    assert real["md"].temp == 330 and real["md"].nsteps == 4000 and real["md"].ps == pytest.approx(4)
    assert real["npt"].temp == 300


# --- run.sh: the job as an editable script --------------------------------------------------------------------------

def test_run_script_round_trip_and_inputs():
    from compute_bot.recipes.md import export as E
    plan = M.plan_stages(params())
    stages = [(s.name, s.grompp) for s in plan]
    text = E.run_script(stages, "1")
    assert text.startswith("#!/bin/bash\n") and "ions.gro" in text.split("set -euo")[0]
    assert "wyślij ten folder jako zip" in text
    assert "MD_MAXWARN=1 md_step npt -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top\n" in text
    assert E.parse_script(text) == stages
    assert E.script_inputs(stages) == ["em.mdp", "ions.gro", "topol.top", "nvt.mdp", "npt.mdp", "md.mdp"]
    short = [(s.name, s.grompp) for s in M.plan_stages(params(nvt_ps=0, npt_ps=0))]
    assert E.script_inputs(short) == ["em.mdp", "ions.gro", "topol.top", "md.mdp"]


def test_export_imports_without_numpy():
    code = ("import sys\nfor m in ('numpy', 'MDAnalysis', 'gemmi', 'matplotlib'): sys.modules[m] = None\n"
            "from compute_bot.recipes.md.export import script_files\nprint(script_files('.'))")
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=60,
                          env=dict(os.environ, PYTHONPATH=str(REPO)))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]"


def _gromacs_job_with_fake_gmx(tmp_path, p):
    from compute_bot.recipes.md import pipeline as P
    gdir = tmp_path / "gromacs"
    gdir.mkdir()
    (gdir / "clean.pdb").write_text("ATOM\n")
    env = G.gmx_env(2)
    plan = M.plan_stages(p)
    P.build_system(gdir, p, env)
    P.write_mdps(gdir, plan, p)
    P.write_script(gdir, plan, env)
    return gdir, plan, env


def test_run_script_reproduces_the_recipe_calls(tmp_path, fake_gmx, monkeypatch):
    from compute_bot.recipes.md import export as E
    from compute_bot.recipes.md import pipeline as P
    monkeypatch.setenv("MD_MAXWARN", "2")
    p = params(nvt_ps=2, npt_ps=2, ns=0.004, mdp="npt:tau_p=5")
    gdir, plan, env = _gromacs_job_with_fake_gmx(tmp_path, p)
    built = len(_calls(fake_gmx))
    P.run_stages(gdir, plan, env)
    recipe_calls = _calls(fake_gmx)[built:]
    assert [c.split()[0] for c in recipe_calls] == ["grompp", "mdrun"] * 4
    assert "-maxwarn 2" in recipe_calls[0]
    files = E.script_files(gdir)
    assert files == ["em.mdp", "ions.gro", "md.mdp", "npt.mdp", "nvt.mdp", "posre.itp", "run.sh", "topol.top"]
    fresh = tmp_path / "zip"
    fresh.mkdir()
    for name in files:
        shutil.copy2(gdir / name, fresh / name)
    fake_gmx.write_text("")
    monkeypatch.delenv("MD_MAXWARN")  # the script carries the value itself
    proc = subprocess.run(["bash", "run.sh"], cwd=fresh, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _calls(fake_gmx) == recipe_calls
    # rerun (the bot's restart): every stage is skipped
    fake_gmx.write_text("")
    again = subprocess.run(["bash", "run.sh"], cwd=fresh, capture_output=True, text=True, timeout=120)
    assert again.returncode == 0 and _calls(fake_gmx) == [] and "md już policzony" in again.stdout
    # the script is rewritten identically on a rerun of the recipe (idempotent)
    before = (gdir / "run.sh").read_text()
    P.write_script(gdir, plan, dict(env, MD_MAXWARN="2"))
    assert (gdir / "run.sh").read_text() == before


def test_script_files_without_script(tmp_path):
    from compute_bot.recipes.md import export as E
    (tmp_path / "md.mdp").write_text("x")
    assert E.script_files(tmp_path) == []


@pytest.mark.slow
def test_gromacs_run_script_1l2y_edited_and_rerun(tmp_path):
    pytest.importorskip("gemmi")
    from compute_bot.recipes.md import export as E
    env = _recipe_env()
    if not shutil.which("gmx", path=env["PATH"]):
        pytest.skip("gmx not on PATH")
    cmd = [sys.executable, "-m", "compute_bot.recipes.md"]
    work = tmp_path / "work"
    work.mkdir()
    p = RECIPES["md"].validate({"pdb": "1L2Y", "nvt_ps": "2", "npt_ps": "2", "ns": "0.004", "seed": "9",
                                "T_eq": "310", "mdp": "nvt:tau_t=0.2 0.2;npt:tau_p=4;md:nstcalcenergy=50"})
    (work / "params.json").write_text(json.dumps(p))
    t0 = time.time()
    run = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=1800)
    t_job = time.time() - t0
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    gdir = work / "gromacs"
    summary = (work / "summary.txt").read_text()
    assert "NVT 2 ps (310 K" in summary and "NPT 2 ps (310 K" in summary and "w 300 K" in summary
    assert "NVT: tau_t=0.2 0.2" in summary and "produkcja: nstcalcenergy=50" in summary
    assert M.parse_mdp((gdir / "npt.mdp").read_text())["tau_p"] == "4"
    files = E.script_files(gdir)
    print("\nscript_files:", files)
    assert {"run.sh", "ions.gro", "topol.top", "posre.itp", "em.mdp", "nvt.mdp", "npt.mdp", "md.mdp"} <= set(files)
    assert not [f for f in files if f.endswith((".xtc", ".trr", ".cpt", ".edr", ".tpr", ".log"))]

    fresh = tmp_path / "zip"
    fresh.mkdir()
    for name in files:
        shutil.copy2(gdir / name, fresh / name)
    md_mdp = (fresh / "md.mdp").read_text()
    assert "nsteps" in md_mdp and "= 2000\n" in md_mdp
    (fresh / "md.mdp").write_text(md_mdp.replace("= 2000\n", "= 1000\n", 1))
    assert M.parse_mdp((fresh / "md.mdp").read_text())["nsteps"] == "1000"
    t1 = time.time()
    rerun = subprocess.run(["bash", "run.sh"], cwd=fresh, env=env, capture_output=True, text=True, timeout=1800)
    t_script = time.time() - t1
    assert rerun.returncode == 0, rerun.stdout[-3000:] + rerun.stderr[-3000:]
    log = (fresh / "md.log").read_text()
    assert "Finished mdrun" in log and re.search(r"^\s*nsteps\s*=\s*1000\s*$", log, re.M)
    assert re.search(r"^\s*nsteps\s*=\s*2000\s*$", (gdir / "md.log").read_text(), re.M)
    for stage in ("em", "nvt", "npt", "md"):
        assert (fresh / f"{stage}.gro").exists()
    print(f"timing: job {t_job:.0f} s, edited run.sh {t_script:.0f} s")
