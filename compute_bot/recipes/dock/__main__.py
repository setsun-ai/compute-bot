import os
import sys

# Thread limits before numpy/OpenMM/Vina are imported (the laptop has 2 cores / 4 threads).
_threads = os.environ.get("COMPUTE_THREADS") or str(os.cpu_count() or 1)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENMM_CPU_THREADS"):
    os.environ.setdefault(_var, _threads)

from compute_bot.recipes.dock.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
