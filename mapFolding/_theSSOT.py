"""Primary goal: access and configure package settings and metadata.

Secondary goal: store hardcoded values until I implement a dynamic solution.
"""

from collections.abc import Callable
from hunterMakesPy import PackageSettings
from pathlib import Path
from typing import TypedDict
import dataclasses
import random

# TODO eliminate hardcoding
concurrencyPackageHARDCODED = 'multiprocessing'
"""Default package identifier for concurrent execution operations."""

class MetadataOEISidManuallySet(TypedDict):
	"""Settings that are best selected by a human instead of algorithmically."""

	getMapShape: Callable[[int], tuple[int, ...]]
	"""Function to convert the OEIS sequence index, 'n', to its `mapShape` tuple."""
	valuesBenchmark: list[int]
	"""List of index values, 'n', to use when benchmarking the algorithm performance."""
	valuesTestParallelization: list[int]
	"""List of index values, 'n', to use when testing parallelization performance."""
	valuesTestValidation: list[int]
	"""List of index values, 'n', to use when testing validation performance."""

@dataclasses.dataclass
class mapFoldingPackageSettings(PackageSettings):
	"""Centralized configuration container for all package-wide settings.

	(AI generated docstring)

	This dataclass serves as the single source of truth for package configuration,
	providing both static and dynamically-resolved values needed throughout the
	package lifecycle. The metadata on each field indicates when that value is
	determined - either during packaging or at installation/runtime.

	The design supports different evaluation phases to optimize performance and
	reliability. Packaging-time values can be determined during package creation,
	while installing-time values require filesystem access or module introspection.

	Attributes
	----------
	fileExtension : str = '.py'
		Standard file extension for Python modules in this package.
	packageName : str
		Canonical name of the package as defined in project configuration.
	pathPackage : Path
		Absolute filesystem path to the installed package directory.
	concurrencyPackage : str | None = None
		Package identifier for concurrent execution operations.

	"""

	concurrencyPackage: str | None = None
	"""
	Package identifier for concurrent execution operations.

	(AI generated docstring)

	Specifies which Python package should be used for parallel processing
	in computationally intensive operations. When None, the default concurrency
	package specified in the module constants is used. Accepted values include
	'multiprocessing' for standard parallel processing and 'numba' for
	specialized numerical computations.
	"""

identifierPackageFALLBACK = "mapFolding"
"""Hardcoded package name used as fallback when dynamic resolution fails."""

concurrencyPackage = concurrencyPackageHARDCODED
"""Active concurrency package configuration for the current session."""

# TODO I made a `TypedDict` before I knew how to make dataclasses and classes. Think about other data structures.
settingsOEISManuallySelected: dict[str, MetadataOEISidManuallySet] = {
	'A000136': {
		'getMapShape': lambda n: tuple(sorted([1, n])),
		'valuesBenchmark': [14],
		'valuesTestParallelization': [*range(3, 7)],
		'valuesTestValidation': [random.randint(2, 9)],  # noqa: S311
	},
	'A001415': {
		'getMapShape': lambda n: tuple(sorted([2, n])),
		'valuesBenchmark': [14],
		'valuesTestParallelization': [*range(3, 7)],
		'valuesTestValidation': [random.randint(2, 9)],  # noqa: S311
	},
	'A001416': {
		'getMapShape': lambda n: tuple(sorted([3, n])),
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
}

packageSettings = mapFoldingPackageSettings(
	identifierPackageFALLBACK=identifierPackageFALLBACK
	, concurrencyPackage=concurrencyPackage)
"""Global package settings."""
cacheDays = 30
"""Number of days to retain cached OEIS data before refreshing from the online source."""
pathCache: Path = packageSettings.pathPackage / ".cache"
"""Local directory path for storing cached OEIS sequence data and metadata."""
