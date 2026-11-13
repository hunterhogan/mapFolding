from __future__ import annotations

from functools import partial
from hunterMakesPy import PackageSettings
from mapFolding.theSSOT import settingsPackage
from pathlib import Path
from sympy import S, Symbol
from sympy.core.function import UndefinedFunction
from typing import TYPE_CHECKING
import dataclasses
import socket

if TYPE_CHECKING:
	from mapFolding.theTypes import OEISid
	from sympy.core.function import AppliedUndef
	from sympy.sets.sets import Set
	from typing import LiteralString

# TODO Dynamic, but authoritative, discovery of OEIS IDs implemented in mapFolding.
# TODO Dynamic, but authoritative, discovery of valid parameter values for `f` and `flow`: primarily for testing.

@dataclasses.dataclass
class PackageSettingsOEIS(PackageSettings):
	urlRepository: str = 'https://github.com/oeis/oeisdata.git'
	"""Git export of the OEIS: sequence entries in `seq/`, and supporting files, such as b-files, in `files/` through Git LFS."""
	pathClone: Path = dataclasses.field(default_factory=Path)
	"""Local sparse clone of `urlRepository`. Nothing commits to this clone."""
	pathSnapshot: Path = dataclasses.field(init=False)
	"""Copy, in the oeisdata directory layout, of the files of `pathClone` for `boxOfIDs`, distributed with the package."""
	pathTerms: Path = dataclasses.field(init=False)
	"""B-files, in the oeisdata directory layout, of terms computed by this package that oeisdata lacks."""

	boxOfIDsMapFolding: tuple[LiteralString, ...] = ('A000136', 'A001415', 'A001416', 'A001417', 'A195646', 'A001418')
	"""OEIS IDs of multidimensional map-folding algorithms."""
	boxOfIDsCounting: tuple[LiteralString, ...] = dataclasses.field(init=False)
	"""OEIS IDs of counting algorithms: map folding, meanders, and symmetric folding."""
	boxOfIDs: tuple[LiteralString, ...] = dataclasses.field(init=False)
	"""Every implemented OEIS ID."""

	n: Symbol = dataclasses.field(default_factory=partial(Symbol, 'n', integer=True, nonnegative=True))
	k: Symbol = dataclasses.field(default_factory=partial(Symbol, 'k', integer=True, nonnegative=True))
	domain: Set = S.Naturals0
	lookupSymbolic: dict[OEISid, type[AppliedUndef]] = dataclasses.field(init=False)
	lookupID: dict[type[AppliedUndef], OEISid] = dataclasses.field(init=False)

	def __post_init__(self, identifierPackageFALLBACK: str) -> None:
		super().__post_init__(identifierPackageFALLBACK)
		if self.pathClone == Path():
			if socket.gethostname() == 'duda':
				self.pathClone = Path('/clones/oeisdata')
			else:
				self.pathClone = self.pathPackage.parents[2] / 'oeisdata'
		self.pathSnapshot = self.pathPackage / 'oeisdata'
		self.pathTerms = self.pathPackage / 'terms'
		self.boxOfIDsCounting = (*self.boxOfIDsMapFolding, 'A000682', 'A005316', 'A007822')
		self.boxOfIDs = (*self.boxOfIDsMapFolding
			, 'A000108', 'A000560', 'A000682', 'A001010', 'A001011', 'A001246', 'A001263', 'A002415'
			, 'A005315', 'A005316', 'A006657', 'A006658', 'A006659', 'A006660', 'A006661', 'A006662', 'A007746', 'A007822', 'A008828'
			, 'A014137', 'A014140', 'A045445', 'A053121', 'A060206', 'A060972', 'A076876'
			, 'A077014', 'A077054', 'A077055', 'A077056', 'A077460', 'A078591', 'A078592', 'A085973'
			, 'A204352', 'A208357', 'A217310', 'A217318', 'A223093', 'A223094', 'A223095'
			, 'A227167', 'A259689', 'A259698', 'A259700', 'A259701', 'A259702', 'A259703', 'A259974', 'A260785', 'A287548'
			, 'A301620', 'A322507', 'A331499', 'A333971', 'A334615', 'A337581', 'A339179', 'A380367', 'A380369'
			, 'A400429', 'A400630', 'A400631', 'A400632', 'A400633', 'A400634', 'A400635', 'A400636', 'A400637', 'A400638'
			, 'A401693', 'A401694'
		)
		self.lookupSymbolic = dict(zip(self.boxOfIDs
			, map(partial(UndefinedFunction, integer=True, nonnegative=True), self.boxOfIDs), strict=True)
		)
		self.lookupID = dict(zip(self.lookupSymbolic.values(), self.lookupSymbolic, strict=True))

settingsOEIS: PackageSettingsOEIS = PackageSettingsOEIS(identifierPackage='mapFolding.oeis', pathPackage=settingsPackage.pathPackage / 'oeis')
