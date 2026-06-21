#!/usr/bin/env python3
"""
Deprecated entry point — implementation lives in tfbd.py.

If your Kaggle notebook still pastes an old million_brains_dflash.py cell that
mentions bitsandbytes / strategy 0, replace it with tfbd.py from this repo.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_TARGET = _ROOT / "tfbd.py"

if __name__ == "__main__":
    print(
        "[DEPRECATED] million_brains_dflash.py delegates to tfbd.py "
        "(HF explicit device_map — no 4-bit quant)."
    )
    sys.argv[0] = str(_TARGET)
    runpy.run_path(str(_TARGET), run_name="__main__")