"""
Foundation layer for the map folding computational ecosystem.

(AI generated docstring)

This module establishes the fundamental configuration infrastructure that underpins
all map folding operations. Map folding, as defined by Lunnon's 1971 algorithm,
requires precise coordination of computational resources, type systems, and data
flow management to solve the complex combinatorial problem of counting distinct
folding patterns across multi-dimensional maps.

The Single Source Of Truth (SSOT) principle governs this foundation, ensuring that
package identity, filesystem locations, and concurrency configuration remain
consistent across all computational phases. During packaging, static metadata is
resolved from pyproject.toml. During installation, filesystem-dependent paths are
dynamically discovered. During runtime, the `packageSettings` instance provides
unified access to all configuration values, enabling the sophisticated computational
framework that follows.

This configuration foundation supports the type system definition, core utility
functions, computational state management, result persistence, and ultimately the
main computational interface that users interact with to solve map folding problems.
"""

from hunterMakesPy import PackageSettings
import dataclasses

packageNameHARDCODED = "mapFolding"
"""Hardcoded package name used as fallback when dynamic resolution fails."""

concurrencyPackageHARDCODED = 'multiprocessing'
"""Default package identifier for concurrent execution operations."""

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

concurrencyPackage = concurrencyPackageHARDCODED
"""Active concurrency package configuration for the current session."""

packageSettings = mapFoldingPackageSettings(
	identifierPackageFALLBACK=packageNameHARDCODED
	, concurrencyPackage=concurrencyPackage)
"""Global package settings."""
