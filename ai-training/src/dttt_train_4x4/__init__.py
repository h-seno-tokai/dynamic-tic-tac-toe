"""DTTT 4x4-only AlphaZero training package.

This package replaces the universal (3x3 + 4x4) training stack with a network
specialised for ``PRESET_4X4_XL`` only.  3x3 has moved to a dedicated exact
solver path (see ``solver_3x3``).

Sibling modules ``rules.py``, ``engine.py`` and ``encoding.py`` are owned by
the parallel "fast 4x4 engine" agent.  This package contributes only the
network and ONNX export glue (``network.py``, ``export_onnx.py``).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
