"""Access and configure package settings and metadata."""

from hunterMakesPy import PackageSettings
from mapFolding import MetadataOEISidManuallySet, MetadataOEISidMapFoldingManuallySet
from more_itertools import loops
from pathlib import Path
import dataclasses

@dataclasses.dataclass
class mapFoldingPackageSettings(PackageSettings):
	"""Widely used settings that are especially useful for map folding algorithms.

	Attributes
	----------
	identifierPackageFALLBACK : str = ''
		Fallback package identifier used only during initialization when automatic discovery fails.
	pathPackage : Path = Path()
		Absolute path to the installed package directory. Automatically resolved from `identifierPackage` if not provided.
	identifierPackage : str = ''
		Canonical name of the package. Automatically extracted from `pyproject.toml`.
	fileExtension : str = '.py'
		Default file extension.

	cacheDays : int = 30
		Number of days to retain cached OEIS data before refreshing from the online source.
	concurrencyPackage : str = 'multiprocessing'
		Package identifier for concurrent execution operations.
	OEISidMapFoldingManuallySet : dict[str, MetadataOEISidMapFoldingManuallySet]
		Settings that are best selected by a human instead of algorithmically.
	OEISidManuallySet : dict[str, MetadataOEISidMeandersManuallySet]
		Settings that are best selected by a human instead of algorithmically for meander sequences.
	"""

	OEISidMapFoldingManuallySet: dict[str, MetadataOEISidMapFoldingManuallySet] = dataclasses.field(default_factory=dict[str, MetadataOEISidMapFoldingManuallySet])
	"""Settings that are best selected by a human instead of algorithmically."""

	OEISidManuallySet: dict[str, MetadataOEISidManuallySet] = dataclasses.field(default_factory=dict[str, MetadataOEISidManuallySet])
	"""Settings that are best selected by a human instead of algorithmically for meander sequences."""

	cacheDays: int = 30
	"""Number of days to retain cached OEIS data before refreshing from the online source."""

	concurrencyPackage: str = 'multiprocessing'
	"""Package identifier for concurrent execution operations."""
# TODO I made this a `TypedDict` before I knew how to make dataclasses and classes. Think about other data structures.
OEISidMapFoldingManuallySet: dict[str, MetadataOEISidMapFoldingManuallySet] = {
	'A000136': {'getMapShape': lambda n: (1, n)},
	'A001415': {'getMapShape': lambda n: (2, n)},
	'A001416': {'getMapShape': lambda n: (3, n)},
	'A001417': {'getMapShape': lambda n: tuple(2 for _dimension in loops(n))},
	'A195646': {'getMapShape': lambda n: tuple(3 for _dimension in loops(n))},
	'A001418': {'getMapShape': lambda n: (n, n)},
}

identifierPackageFALLBACK = "mapFolding"
"""Manually entered package name used as fallback when dynamic resolution fails."""

packageSettings = mapFoldingPackageSettings(identifierPackageFALLBACK=identifierPackageFALLBACK, OEISidMapFoldingManuallySet=OEISidMapFoldingManuallySet)
"""Global package settings."""

OEISidManuallySet: dict[str, MetadataOEISidManuallySet] = {'A000560': {}, 'A000682': {}, 'A001010': {}, 'A001011': {},
	'A005315': {}, 'A005316': {}, 'A007822': {}, 'A060206': {}, 'A077460': {}, 'A078591': {},
	'A086345': {}, 'A178961': {}, 'A223094': {}, 'A259702': {}, 'A301620': {}}

# Recreate packageSettings with meanders settings included
packageSettings = mapFoldingPackageSettings(
	identifierPackageFALLBACK=identifierPackageFALLBACK,
	OEISidMapFoldingManuallySet=OEISidMapFoldingManuallySet,
	OEISidManuallySet=OEISidManuallySet,
)
"""Global package settings."""

# TODO integrate into packageSettings
pathCache: Path = packageSettings.pathPackage / ".cache"
"""Local directory path for storing cached OEIS sequence data and metadata."""
