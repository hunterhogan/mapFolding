"""Display the registered OEIS sequences and usage examples when you run `python -m mapFolding.oeis`.

(AI generated docstring)

Executing the package displays sequence identifiers, descriptions, and usage examples through
`mapFolding.oeis.getOEISids`. The installed `getOEISids` console command provides the same listing.
"""
from __future__ import annotations

from mapFolding.oeis._commandLine import getOEISids

if __name__ == "__main__":
	getOEISids()
