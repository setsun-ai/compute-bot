#!/usr/bin/env python3
"""Starts the bot or the runner from the newest copy: the installed one or the one the bot downloaded.
   launch.py bot | runner"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compute_bot.updater import chosen_app  # noqa: E402

if len(sys.argv) < 2 or sys.argv[1] not in ("bot", "runner"):
    sys.exit("użycie: launch.py bot|runner")
app = chosen_app(HERE, Path(os.environ.get("COMPUTE_HOME", ".")) / "update")
os.environ["PYTHONPATH"] = str(app)
os.environ["COMPUTE_INSTALLED"] = str(HERE)
os.chdir(app)
os.execv(sys.executable, [sys.executable, "-m", f"compute_bot.{sys.argv[1]}", *sys.argv[2:]])
