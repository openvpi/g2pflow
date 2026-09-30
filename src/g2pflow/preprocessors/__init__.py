from pathlib import Path

from ..discovery import _discover_package
from .base import Preprocessor


_discover_package(__name__, Path(__file__).parent, builtin=True)
