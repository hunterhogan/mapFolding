"""Display available OEIS sequence information when executed as a module.

(AI generated docstring)

You can execute this module with `python -m mapFolding.oeis` to display the identifiers,
descriptions, and usage examples for implemented OEIS sequences. The module delegates output
generation to `getOEISids` [1].

Contents
--------
Functions
	getOEISids
		Display comprehensive information about all implemented OEIS sequences.

References
----------
[1] `mapFolding.oeis.getOEISids`
	Internal package reference for the OEIS sequence information display function.
"""
from __future__ import annotations

from mapFolding.oeis import getOEISids

getOEISids()
