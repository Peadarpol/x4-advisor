"""Entrypoint wrapper for running M6 model bake-off benchmark."""

import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_eval_benchmark import main

if __name__ == "__main__":
    main()
