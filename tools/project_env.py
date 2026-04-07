"""Load the repository root `.env` so `os.environ` has API keys before use."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# tools/project_env.py -> parents[1] == repo root
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_project_env() -> None:
    """Load `.env` from the project root (independent of current working directory)."""
    load_dotenv(_PROJECT_ROOT / ".env")
