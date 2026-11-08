"""python3 -m compute_bot --selftest | --version: checks that this copy can run here (the updater uses it)."""
import os
import sys

from . import VERSION


def selftest():
    try:
        from . import bot, jobs, runner, telegram, updater  # noqa: F401  everything imports on this Python
        from .recipes import RECIPES
        helper = os.path.join(os.path.dirname(__file__), "bin", "md_step")
        if not os.access(helper, os.X_OK):
            raise RuntimeError("md_step is not executable")
        for recipe in RECIPES.values():
            recipe.validate({})
    except Exception as e:
        print(f"Selftest failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"OK {VERSION}")
    return 0


if __name__ == "__main__":
    if "--version" in sys.argv:
        print(VERSION)
        sys.exit(0)
    sys.exit(selftest())
