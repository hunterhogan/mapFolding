"""Access and configure package settings and metadata."""

from hunterMakesPy import PackageSettings
from mapFolding._theTypes import MetadataOEISidManuallySet, MetadataOEISidMapFoldingManuallySet
from pathlib import Path
import dataclasses
import random

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
# ruff: noqa: S311
# TODO I made this a `TypedDict` before I knew how to make dataclasses and classes. Think about other data structures.
OEISidMapFoldingManuallySet: dict[str, MetadataOEISidMapFoldingManuallySet] = {
	'A000136': {
		'getMapShape': lambda n: (1, n),
		'valuesBenchmark': [14],
		'valuesTestParallelization': [random.randint(3, 7)],
		'valuesTestValidation': [random.randint(2, 9)],
	},
	'A001415': {
		'getMapShape': lambda n: (2, n),
		'valuesBenchmark': [14],
		'valuesTestParallelization': [random.randint(3, 7)],
		'valuesTestValidation': [random.randint(2, 9)],
	},
	'A001416': {
		'getMapShape': lambda n: (3, n),
		'valuesBenchmark': [9],
		'valuesTestParallelization': [random.randint(3, 5)],
		'valuesTestValidation': [random.randint(2, 6)],
	},
	'A001417': {
		'getMapShape': lambda n: tuple(2 for _dimension in range(n)),
		'valuesBenchmark': [6],
		'valuesTestParallelization': [random.randint(2, 4)],
		'valuesTestValidation': [random.randint(2, 4)],
	},
	'A195646': {
		'getMapShape': lambda n: tuple(3 for _dimension in range(n)),
		'valuesBenchmark': [3],
		'valuesTestParallelization': [2],
		'valuesTestValidation': [2],
	},
	'A001418': {
		'getMapShape': lambda n: (n, n),
		'valuesBenchmark': [5],
		'valuesTestParallelization': [*range(2, 4)],
		'valuesTestValidation': [random.randint(2, 4)],
	},
}

identifierPackageFALLBACK = "mapFolding"
"""Manually entered package name used as fallback when dynamic resolution fails."""

packageSettings = mapFoldingPackageSettings(identifierPackageFALLBACK=identifierPackageFALLBACK, OEISidMapFoldingManuallySet=OEISidMapFoldingManuallySet)
"""Global package settings."""

OEISidManuallySet: dict[str, MetadataOEISidManuallySet] = {
	'A000560': {'valuesTestValidation': [random.randint(3, 12)]},
	'A000682': {'valuesTestValidation': [random.randint(3, 12), 32]},
	'A001010': {'valuesTestValidation': [3, 4, random.randint(5, 11)]},
	'A001011': {'valuesTestValidation': [3, 4, random.randint(5, 7)]},
	'A005315': {'valuesTestValidation': [random.randint(3, 9)]},
	'A005316': {'valuesTestValidation': [random.randint(3, 13)]},
	'A007822': {'valuesTestValidation': [random.randint(2, 8)]}, #, 'valuesBenchmark': [7], 'valuesTestParallelization': [*range(2, 4)]},
	'A060206': {'valuesTestValidation': [random.randint(3, 9)]},
	'A077460': {'valuesTestValidation': [3, 4, random.randint(5, 8)]},
	'A078591': {'valuesTestValidation': [random.randint(3, 10)]},
	'A086345': {'valuesTestValidation': [random.randint(3, 10), random.randint(11, 20), random.randint(21, 30), random.randint(31, 40)]},
	'A178961': {'valuesTestValidation': [random.randint(3, 11)]},
	'A223094': {'valuesTestValidation': [random.randint(3, 11)]},
	'A259702': {'valuesTestValidation': [random.randint(3, 13)]},
	'A301620': {'valuesTestValidation': [random.randint(3, 11)]},
}

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
