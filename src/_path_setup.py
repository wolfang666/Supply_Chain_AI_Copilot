"""
_path_setup.py
──────────────
Import this module first in any file that needs to resolve sibling packages.
It adds `src/` to sys.path so that `ui`, `config`, `analytics`, etc.
are all importable as top-level names — on Windows and Unix alike.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent          
_ROOT = _SRC.parent                             
_PAGES = _ROOT / "pages"

for _p in (str(_SRC), str(_PAGES), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
