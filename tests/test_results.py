"""What goes in the results zip when not everything fits, and what can go to save disk."""
from compute_bot import results

MB = 1024 ** 2


def test_everything_goes_in_when_it_fits():
    files = [("run.gro", 600_000), ("wyniki/a.csv", 10), ("md.cpt", 5000)]
    included, left = results.pack(files, budget=10 * MB)
    assert {r for r, _ in included} == {"run.gro", "wyniki/a.csv"} and left == []


def test_when_it_does_not_fit_results_come_first_and_intermediates_stay():
    files = [(f"runs/r{i:02d}/run.gro", 600_000) for i in range(100)]
    files += [("wyniki/table.csv", 2_000), ("wyniki/plot.png", 300_000), ("summary.txt", 100),
              ("runs/r01/rmsd.xvg", 50_000), ("runs/r01/md.cpt", 400_000)]
    included, left = results.pack(files, budget=10 * MB, highlights=["runs/r01/rmsd.xvg"])
    names = {r for r, _ in included}
    assert {"wyniki/table.csv", "wyniki/plot.png", "summary.txt", "runs/r01/rmsd.xvg"} <= names
    assert not any(r.endswith("run.gro") for r in names)
    assert len(left) == 100  # the gro files; the checkpoint is neither sent nor listed


def test_the_most_important_results_win_when_even_they_do_not_fit():
    files = [("wyniki/big.png", 6 * MB), ("other.csv", 5 * MB), ("summary.txt", 10)]
    included, left = results.pack(files, budget=8 * MB)
    assert [r for r, _ in included] == ["summary.txt", "wyniki/big.png"] and left == [("other.csv", 5 * MB)]


def test_what_slimming_removes():
    files = ["md.cpt", "nvt_prev.cpt", "#topol.top.1#", "core.1234", "step3b.pdb", "wyniki/a.csv",
             "md.xtc", "md.tpr", "topol.top"]
    assert sorted(results.slimmable(files)) == ["#topol.top.1#", "core.1234", "md.cpt", "nvt_prev.cpt",
                                                 "step3b.pdb"]


def test_copy_command_for_a_terminal():
    command = results.copy_command("vostro", "/var/lib/compute-bot/jobs/0004/work", "~/md/met", 4)
    assert command == ("mkdir -p ~/md/met/bot_0004 && rsync -av --exclude '*.cpt' --exclude '#*#' "
                       "vostro:/var/lib/compute-bot/jobs/0004/work/ ~/md/met/bot_0004/")
