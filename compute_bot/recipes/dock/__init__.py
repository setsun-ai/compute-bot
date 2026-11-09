"""
Docking recipe (AutoDock Vina): receptor prep (pdbfixer + Meeko), box, ligand prep (RDKit + Open Babel + Meeko),
resumable virtual screening one ligand at a time, ranking, poses, contacts and a Polish summary.

Run as `python -m compute_bot.recipes.dock` in the job's work directory (see docs/RECIPE_CONTRACT.md).
The submodules keep the computation in pure functions; file input/output lives in `main.py`.
"""
