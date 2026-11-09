"""Proteomics / metabolomics recipe: intensity matrix + sample sheet -> QC, normalisation, statistics, annotation.

Started as `python -m compute_bot.recipes.omics` (see `__main__.py`). The steps live in pure functions:
`parse` (input shapes), `prep` (QC, filtering, imputation, normalisation, drift and batch correction),
`stats` (tests, PCA, PLS-DA), `annotate` (UniProt, g:Profiler), `plots`, and `pipeline` that chains them and
records what each step did.
"""
