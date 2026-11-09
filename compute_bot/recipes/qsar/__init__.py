"""QSAR / QSPR recipe: table → descriptors → pre-filter → split → variable selection → models → OECD validation.

Kept free of imports so that `python -m compute_bot.recipes.qsar` can set thread limits before numpy loads.
"""
