"""Small shared helpers: the owner-facing error type and progress lines."""
from __future__ import annotations

import sys


class UserError(Exception):
    """Something in the owner's input that they have to fix; the message is Polish and says what to do."""


def progress(text: str) -> None:
    print(f"PROGRESS: {text}", flush=True)


def log(text: str) -> None:
    print(text, flush=True)


def warn(text: str) -> None:
    print(f"UWAGA: {text}", file=sys.stderr, flush=True)
