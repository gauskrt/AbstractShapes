"""AbstractShapes: reproduce images with geometric primitives.

A Python port of Michael Fogleman's `primitive`
(https://github.com/fogleman/primitive).
"""

from .model import Model
from .shapes import MODE_NAMES, MODES

__all__ = ["Model", "MODES", "MODE_NAMES"]
__version__ = "0.1.0"
