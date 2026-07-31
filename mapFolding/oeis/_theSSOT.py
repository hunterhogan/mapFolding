from __future__ import annotations

from mapFolding.theSSOT import settingsPackage
from more_itertools import loops
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding.oeis._dataBaskets import MetadataOEISidManuallySet, MetadataOEISidMapFoldingManuallySet
	from pathlib import Path

# TODO merge dictionaries
# TODO refactor to use `getMapShape` function instead of `oeisIDsImplementedMapFolding` dictionary.
# TODO Dynamic, but authoritative, discovery of OEIS IDs implemented in mapFolding.
# TODO Dynamic, but authoritative, discovery of valid parameter values for `f` and `flow`: primarily for testing.

oeisIDsImplementedMapFolding: dict[str, MetadataOEISidMapFoldingManuallySet] = {
	'A000136': {'getMapShape': lambda n: (1, n)}
	, 'A001415': {'getMapShape': lambda n: (2, n)}
	, 'A001416': {'getMapShape': lambda n: (3, n)}
	, 'A001417': {'getMapShape': lambda n: tuple(2 for _dimension in loops(n))}
	, 'A195646': {'getMapShape': lambda n: tuple(3 for _dimension in loops(n))}
	, 'A001418': {'getMapShape': lambda n: (n, n)}
}
"""Settings that are best selected by a human instead of algorithmically."""

oeisIDsImplemented: dict[str, MetadataOEISidManuallySet] = {'A000560': {}, 'A000682': {}, 'A001010': {}, 'A001011': {}
	, 'A005315': {}, 'A005316': {}, 'A007822': {}, 'A060206': {}, 'A077014': {}, 'A077054': {}
	, 'A077460': {}, 'A078591': {}, 'A078592': {}, 'A085973': {}, 'A208357': {}, 'A217310': {}
	, 'A217318': {}, 'A223093': {}, 'A223094': {}, 'A223095': {}, 'A227167': {}, 'A259702': {}, 'A301620': {}
	, 'A330269': {}, 'A333971': {}, 'A334615': {}, 'A337581': {}}
"""Settings that are best selected by a human instead of algorithmically for meander sequences."""

cacheDays: int = 30
"""Number of days to retain cached OEIS data before refreshing from the online source."""

pathCache: Path = settingsPackage.pathPackage / "oeis" / ".cache"
"""Local directory path for storing cached OEIS sequence data and metadata."""

def getMapShape(oeisID: str, n: int) -> tuple[int, ...]:
	"""Get the map shape for a given OEIS ID and index n."""
	match oeisID:
		case 'A000136':
			mapShape: tuple[int, ...] = (1, n)
		case 'A001415':
			mapShape = (2, n)
		case 'A001416':
			mapShape = (3, n)
		case 'A001417':
			mapShape = tuple(2 for _dimension in loops(n))
		case 'A195646':
			mapShape = tuple(3 for _dimension in loops(n))
		case 'A001418':
			mapShape = (n, n)
		case _:
			message: str = f"I received `{oeisID = }`, but it is not implemented in `getMapShape`."
			raise ValueError(message)
	return mapShape
