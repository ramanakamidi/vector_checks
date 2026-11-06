"""Shared paths for application data stored in the project root."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = PROJECT_ROOT / "projects"
