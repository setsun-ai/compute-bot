"""The guide's "try it" examples must really run and show what their note promises."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from compute_bot import examples
from compute_bot.recipes import RECIPES

REPO = Path(__file__).resolve().parents[2]


def run_example(name, tmp_path, timeout=1800):
    example = examples.EXAMPLES[name]
    for rel, data in example.files().items():
        (tmp_path / rel).write_bytes(data)
    (tmp_path / "params.json").write_text(json.dumps(RECIPES[example.recipe].validate(example.typed)))
    env = dict(os.environ, PYTHONPATH=str(REPO), COMPUTE_THREADS="4", COMPUTE_BOT_OFFLINE="1",
               PATH=os.pathsep.join([str(REPO / "compute_bot/bin"), str(Path(sys.executable).parent),
                                     os.environ["PATH"]]))
    done = subprocess.run([sys.executable, "-m", f"compute_bot.recipes.{example.recipe}"], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=timeout)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    return (tmp_path / "summary.txt").read_text()


def test_qsar_example_finds_the_two_descriptors_it_was_built_from(tmp_path):
    pytest.importorskip("sklearn")
    summary = run_example("qsar", tmp_path)
    assert "logD" in summary and "V" in summary and "Mw" not in summary.split("Wybór")[1].split("\n")[0]


def test_omics_example_finds_its_differences_and_drops_the_background(tmp_path):
    pytest.importorskip("statsmodels")
    run_example("omics", tmp_path)
    stats = next(tmp_path.rglob("wyniki/**/*stat*.csv"))
    import pandas as pd
    table = pd.read_csv(stats)
    fdr = next(c for c in table.columns if c.lower() in ("fdr", "q", "q_value", "padj", "p_adj"))
    ids = table.columns[0]
    significant = set(table.loc[table[fdr] < 0.05, ids])
    truth = {f"F{i:03d}" for i in range(1, 13)}
    assert len(truth & significant) >= 10 and len(significant - truth) <= 3
    assert not any(f"F{i:03d}" in set(table[ids]) for i in range(141, 151))  # blank background removed


@pytest.mark.slow
def test_md_example_runs(tmp_path):
    pytest.importorskip("MDAnalysis")
    assert "1L2Y" in run_example("md", tmp_path)


@pytest.mark.slow
def test_dock_example_puts_biotin_back_and_first(tmp_path):
    pytest.importorskip("vina")
    run_example("dock", tmp_path)
    import pandas as pd
    ranking = pd.read_csv(tmp_path / "wyniki/ranking.csv")
    assert "biot" in str(ranking.iloc[0].to_dict()).lower() or "BTN" in str(ranking.iloc[0].to_dict())
