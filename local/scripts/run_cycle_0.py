#!/usr/bin/env python3
"""Backward-compatible entrypoint for cycle runner.

This module keeps legacy imports working while delegating execution to
`local/scripts/run_cycle.py`.
"""

import sys
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from binary_expert_model import BinaryExpertModel
from run_cycle import BaselineEnsemble, main


if __name__ == "__main__":
    main()
