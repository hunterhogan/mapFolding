"""Access and configure package settings and metadata."""

from hunterMakesPy import PackageSettings
from mapFolding._theTypes import MetadataOEISidMapFoldingManuallySet, MetadataOEISidMeandersManuallySet
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
	OEISidMeandersManuallySet : dict[str, MetadataOEISidMeandersManuallySet]
		Settings that are best selected by a human instead of algorithmically for meander sequences.
	"""

	OEISidMapFoldingManuallySet: dict[str, MetadataOEISidMapFoldingManuallySet] = dataclasses.field(default_factory=dict[str, MetadataOEISidMapFoldingManuallySet])
	"""Settings that are best selected by a human instead of algorithmically."""

	OEISidMeandersManuallySet: dict[str, MetadataOEISidMeandersManuallySet] = dataclasses.field(default_factory=dict[str, MetadataOEISidMeandersManuallySet])
	"""Settings that are best selected by a human instead of algorithmically for meander sequences."""

	cacheDays: int = 30
	"""Number of days to retain cached OEIS data before refreshing from the online source."""

	concurrencyPackage: str = 'multiprocessing'
	"""Package identifier for concurrent execution operations."""

# TODO I made a `TypedDict` before I knew how to make dataclasses and classes. Think about other data structures.
OEISidMapFoldingManuallySet: dict[str, MetadataOEISidMapFoldingManuallySet] = {
	'A000136': {
		'getMapShape': lambda n: (1, n),
		'valuesBenchmark': [14],
		'valuesTestParallelization': [*range(3, 7)],
		'valuesTestValidation': [random.randint(2, 9)],  # noqa: S311
	},
	'A001415': {
		'getMapShape': lambda n: (2, n),
		'valuesBenchmark': [14],
		'valuesTestParallelization': [*range(3, 7)],
		'valuesTestValidation': [random.randint(2, 9)],  # noqa: S311
	},
	'A001416': {
		'getMapShape': lambda n: (3, n),
		'valuesBenchmark': [9],
		'valuesTestParallelization': [*range(3, 5)],
		'valuesTestValidation': [random.randint(2, 6)],  # noqa: S311
	},
	'A001417': {
		'getMapShape': lambda n: tuple(2 for _dimension in range(n)),
		'valuesBenchmark': [6],
		'valuesTestParallelization': [*range(2, 4)],
		'valuesTestValidation': [random.randint(2, 4)],  # noqa: S311
	},
	'A195646': {
		'getMapShape': lambda n: tuple(3 for _dimension in range(n)),
		'valuesBenchmark': [3],
		'valuesTestParallelization': [*range(2, 3)],
		'valuesTestValidation': [2],
	},
	'A001418': {
		'getMapShape': lambda n: (n, n),
		'valuesBenchmark': [5],
		'valuesTestParallelization': [*range(2, 4)],
		'valuesTestValidation': [random.randint(2, 4)],  # noqa: S311
	},
	'A007822': {
		'getMapShape': lambda n: (1, 2 * n),
		'valuesBenchmark': [7],
		'valuesTestParallelization': [*range(2, 4)],
		'valuesTestValidation': [random.randint(2, 8)],  # noqa: S311
	},
}

identifierPackageFALLBACK = "mapFolding"
"""Manually entered package name used as fallback when dynamic resolution fails."""

packageSettings = mapFoldingPackageSettings(identifierPackageFALLBACK=identifierPackageFALLBACK, OEISidMapFoldingManuallySet=OEISidMapFoldingManuallySet)
"""Global package settings."""

# TODO integrate into packageSettings
pathCache: Path = packageSettings.pathPackage / ".cache"
"""Local directory path for storing cached OEIS sequence data and metadata."""
OEISidMeandersManuallySet: dict[str, MetadataOEISidMeandersManuallySet] = {
	'A000560': {'valuesTestValidation': [*range(3, 12)]},
	'A000682': {'valuesTestValidation': [*range(3, 12)]},
	'A001010': {'valuesTestValidation': [*range(3, 11)]},
	'A001011': {'valuesTestValidation': [*range(3, 7)]},
	'A005315': {'valuesTestValidation': [*range(3, 9)]},
	'A005316': {'valuesTestValidation': [*range(3, 13)]},
	'A060206': {'valuesTestValidation': [*range(3, 9)]},
	'A077460': {'valuesTestValidation': [*range(3, 8)]},
	'A078591': {'valuesTestValidation': [*range(3, 10)]},
	'A178961': {'valuesTestValidation': [*range(3, 11)]},
	'A223094': {'valuesTestValidation': [*range(3, 11)]},
	'A259702': {'valuesTestValidation': [*range(3, 13)]},
	'A301620': {'valuesTestValidation': [*range(3, 11)]},
}

# Recreate packageSettings with meanders settings included
packageSettings = mapFoldingPackageSettings(
	identifierPackageFALLBACK=identifierPackageFALLBACK,
	OEISidMapFoldingManuallySet=OEISidMapFoldingManuallySet,
	OEISidMeandersManuallySet=OEISidMeandersManuallySet,
)
"""Global package settings."""
