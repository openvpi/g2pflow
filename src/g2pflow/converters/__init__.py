from pathlib import Path

from ..discovery import _discover_package
from .base import Converter, G2PConversionError, G2PGroup, G2PPath, G2PWord, G2PReading


_discover_package(__name__, Path(__file__).parent, builtin=True)
