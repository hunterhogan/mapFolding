from __future__ import annotations

from functools import partial
from mapFolding.theSSOT import settingsPackage
from sympy import S, Symbol
from sympy.core.function import UndefinedFunction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding.theTypes import OEISid
	from pathlib import Path
	from sympy.sets.sets import Set
	from typing import LiteralString

cacheDays: int = 30
"""Number of days to retain cached OEIS data before refreshing from the online source."""

# TODO Dynamic, but authoritative, discovery of OEIS IDs implemented in mapFolding.
# TODO Dynamic, but authoritative, discovery of valid parameter values for `f` and `flow`: primarily for testing.

oeisIDsMapFoldingImplemented: tuple[LiteralString, ...] = ('A000136', 'A001415', 'A001416', 'A001417', 'A195646', 'A001418')
"""OEIS IDs of multidimensional map-folding algorithms."""

oeisIDsCountingImplemented: tuple[LiteralString, ...] = (*oeisIDsMapFoldingImplemented, 'A000682', 'A005316', 'A007822')
"""OEIS IDs of counting algorithms: map folding, meanders, and symmetric folding."""

oeisIDsImplemented: tuple[LiteralString, ...] = (*oeisIDsMapFoldingImplemented
	, 'A000108', 'A000560', 'A000682', 'A001010', 'A001011'
	, 'A005315', 'A005316', 'A006661', 'A006662', 'A007822'
	, 'A045445', 'A060206', 'A076876', 'A077014', 'A077054', 'A077055', 'A077460', 'A078591', 'A078592', 'A085973'
	, 'A204352', 'A208357', 'A217310', 'A217318', 'A223093', 'A223094', 'A223095'
	, 'A227167', 'A259689', 'A259698', 'A259701', 'A259702', 'A259703', 'A259974', 'A287548'
	, 'A301620', 'A331499', 'A333971', 'A334615', 'A337581', 'A339179', 'A380367'
	, 'A400429', 'A400630', 'A400631', 'A400632', 'A400633', 'A400634', 'A400635', 'A400636', 'A400637'
	#, 'A400638', 'A400639'
	#, 'A401693', 'A401694'
)
"""Every implemented OEIS ID."""
pathCache: Path = settingsPackage.pathPackage / 'oeis' / '.cache'
"""Local directory path for storing cached OEIS sequence data and metadata."""

nSymbolic: Symbol = Symbol('n', integer=True, nonnegative=True)
kSymbolic: Symbol = Symbol('k', integer=True, nonnegative=True)
domainSymbolic: Set = S.Naturals0  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

lookupSymbolic: dict[OEISid, UndefinedFunction] = dict(zip(oeisIDsImplemented
	, map(partial(UndefinedFunction, integer=True, nonnegative=True), oeisIDsImplemented), strict=True)
)

lookupOEISid: dict[UndefinedFunction, OEISid] = dict(zip(lookupSymbolic.values(), lookupSymbolic, strict=True))
lookupColumnStart: dict[OEISid, int] = {'A259689': 2, 'A259698': 1, 'A259701': 1, 'A259703': 1, 'A339179': 2, 'A400429': 1}
