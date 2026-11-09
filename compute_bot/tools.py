"""
Programs that can't come from conda. NAMD is free for research but downloads only after registering at
ks.uiuc.edu, so the owner brings the Linux-x86_64-multicore tarball (to the bot or deploy/install-namd.sh).

  python -m compute_bot.tools namd NAMD_3.0.1_Linux-x86_64-multicore.tar.gz

Installs into COMPUTE_HOME/tools/<folder>/ and links the binary into COMPUTE_HOME/tools/bin, which the runner puts
on every job's PATH. Installing the same tarball again changes nothing. Standard library only.
"""
from __future__ import annotations

import os
import sys
import tarfile
from pathlib import Path, PurePosixPath

BINARIES = ("namd3", "namd2")


def tools_dir():
    return Path(os.environ.get("COMPUTE_HOME", ".")) / "tools"


def install_namd(tarball, tools):
    tools = Path(tools)
    with tarfile.open(tarball) as archive:
        members = archive.getmembers()
        top = {PurePosixPath(m.name).parts[0] for m in members if m.name}
        if len(top) != 1:
            raise ValueError("to nie wygląda na paczkę NAMD (jeden folder NAMD_...)")
        (folder,) = top
        for m in members:
            path = PurePosixPath(m.name)
            if path.is_absolute() or ".." in path.parts or m.isdev() or (m.issym() and m.linkname.startswith("/")):
                raise ValueError(f"niebezpieczna ścieżka w paczce: {m.name}")
        target = tools / folder
        if not target.exists():
            tmp = tools / (folder + ".tmp")
            tmp.mkdir(parents=True, exist_ok=True)
            archive.extractall(tmp, filter="tar") if sys.version_info >= (3, 12) else archive.extractall(tmp)
            os.replace(tmp / folder, target)
            tmp.rmdir()
    binary = next((target / b for b in BINARIES if (target / b).is_file()), None)
    if binary is None:
        raise ValueError("w paczce nie ma namd3 ani namd2")
    bin_dir = tools / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    link = bin_dir / binary.name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(binary)
    return link


def main(argv):
    if len(argv) != 2 or argv[0] != "namd":
        print("ERROR: użycie: python -m compute_bot.tools namd PLIK.tar.gz", file=sys.stderr)
        return 2
    try:
        link = install_namd(argv[1], tools_dir())
    except (OSError, ValueError, tarfile.TarError) as e:
        print(f"ERROR: NAMD się nie zainstalował: {e}", file=sys.stderr)
        return 1
    Path("summary.txt").write_text(f"NAMD zainstalowany: {link.name} → {link.resolve().parent.name}.\n"
                                   "Jest teraz w PATH każdego zadania (np. charmm engine=namd albo run.sh).\n")
    print(f"PROGRESS: gotowe, {link}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
