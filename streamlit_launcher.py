"""Load Torch before Streamlit's CLI initializes on Windows."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import huggingface_hub.file_download

huggingface_hub.file_download.are_symlinks_supported = lambda *args, **kwargs: False

from streamlit.web.cli import main


if __name__ == "__main__":
    app_path = Path(__file__).with_name("app.py")
    sys.argv = [sys.argv[0], "run", str(app_path), *sys.argv[1:]]
    main()