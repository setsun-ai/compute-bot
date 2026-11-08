"""Recipe env: brings the conda environment up to date with deploy/environment.yml (new packages after /update)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[2]


def main():
    prefix = os.environ.get("CONDA_PREFIX") or str(Path(sys.executable).parent.parent)
    tool = shutil.which("micromamba") or shutil.which("mamba") or shutil.which("conda")
    if not tool:
        print("ERROR: nie widzę micromamby/mamby; uruchom instalator jeszcze raz.", file=sys.stderr)
        return 1
    spec = APP / "deploy" / "environment.yml"
    print(f"PROGRESS: aktualizuję {prefix} z {spec.name}", flush=True)
    args = [tool, "env", "update", "-y", "-p", prefix, "-f", str(spec)] if "micromamba" in tool \
        else [tool, "env", "update", "-p", prefix, "-f", str(spec)]
    code = subprocess.call(args)
    Path("summary.txt").write_text("Środowisko zaktualizowane.\n" if code == 0 else f"Aktualizacja nie wyszła ({code}).\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
