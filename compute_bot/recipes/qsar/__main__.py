"""Entry point: `python -m compute_bot.recipes.qsar` in the job's work directory (reads params.json)."""
import os
import sys


def _limit_threads():
    try:
        n = max(1, int(os.environ.get("COMPUTE_THREADS", "")))
    except ValueError:
        n = max(1, min(4, os.cpu_count() or 1))
    # numpy/BLAS threads per process stay at 1: parallelism comes from n_jobs and MOPAC workers
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    return n


def main():
    import json
    import traceback
    import warnings

    threads = _limit_threads()
    from .pipeline import run
    from .table import InputError

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        with open("params.json", encoding="utf-8") as fh:
            params = json.load(fh)
        run(os.getcwd(), params, threads)
    except InputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(2)
    except Exception as exc:  # unexpected: keep the traceback, then the owner-facing last line
        traceback.print_exc()
        print(f"ERROR: nieoczekiwany błąd przepisu qsar ({type(exc).__name__}: {exc}). Zgłoś to z plikiem wejściowym.",
              file=sys.stderr, flush=True)
        sys.exit(1)
    print("PROGRESS: gotowe", flush=True)


if __name__ == "__main__":
    main()
