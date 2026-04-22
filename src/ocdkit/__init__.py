"""
ocdkit — Obsessive Coder's Dependency Toolkit.

Python utilities for array manipulation, GPU dispatch, image I/O,
morphology, and plotting. Designed for use across multiple projects. 
"""

from .load import enable_submodules

enable_submodules(__name__)
