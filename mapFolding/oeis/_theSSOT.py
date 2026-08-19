from __future__ import annotations

from mapFolding.theSSOT import settingsPackage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from pathlib import Path
	from typing import LiteralString

cacheDays: int = 30
"""Number of days to retain cached OEIS data before refreshing from the online source."""

# TODO Dynamic, but authoritative, discovery of OEIS IDs implemented in mapFolding.
# TODO Dynamic, but authoritative, discovery of valid parameter values for `f` and `flow`: primarily for testing.

oeisIDsMapFoldingImplemented: tuple[LiteralString, ...] = ('A000136', 'A001415', 'A001416', 'A001417', 'A195646', 'A001418')
"""OEIS IDs of multidimensional map-folding algorithms."""

oeisIDsImplemented: tuple[LiteralString, ...] = (*oeisIDsMapFoldingImplemented, 'A000560', 'A000682', 'A001010', 'A001011'
	, 'A005315', 'A005316', 'A007822', 'A060206', 'A077014', 'A077054'#, 'A077055'
	, 'A077460', 'A078591'
	, 'A078592', 'A085973', 'A208357', 'A217310', 'A217318', 'A223093', 'A223094', 'A223095'
	, 'A227167', 'A259689', 'A259702', 'A301620', 'A333971', 'A334615', 'A337581')
"""Every implemented OEIS ID."""

pathCache: Path = settingsPackage.pathPackage / "oeis" / ".cache"
"""Local directory path for storing cached OEIS sequence data and metadata."""
