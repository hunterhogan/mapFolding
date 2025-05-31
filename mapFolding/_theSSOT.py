"""
Foundation layer for the map folding computational ecosystem.

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

from importlib import import_module as importlib_import_module
from inspect import getfile as inspect_getfile
from pathlib import Path
from tomli import load as tomli_load
import dataclasses

packageNamePACKAGING_HARDCODED = "mapFolding"
"""
Hardcoded package name used as fallback when dynamic resolution fails.

This constant serves as the ultimate fallback for package name resolution,
ensuring the package can function even when pyproject.toml is not accessible
during packaging or when module introspection fails during installation.
"""

concurrencyPackageHARDCODED = 'multiprocessing'
"""
Default package identifier for concurrent execution operations.

Specifies which Python concurrency package should be used as the default
for parallel computations. This can be overridden through PackageSettings
to use alternative packages like 'numba' for specialized performance scenarios.
"""

# Evaluate When Packaging
# https://github.com/hunterhogan/mapFolding/issues/18
try:
	packageNamePACKAGING: str = tomli_load(Path("../pyproject.toml").open('rb'))["project"]["name"]
	"""
	Package name dynamically resolved from pyproject.toml during packaging.

	This value is determined by reading the project configuration file during
	the packaging process, ensuring consistency between the package metadata
	and runtime identification. Falls back to hardcoded value if resolution fails.
	"""
except Exception:
	packageNamePACKAGING = packageNamePACKAGING_HARDCODED

# Evaluate When Installing
# https://github.com/hunterhogan/mapFolding/issues/18
def getPathPackageINSTALLING() -> Path:
	"""
	Resolve the absolute filesystem path to the installed package directory.

	This function determines the package location at runtime by introspecting
	the imported module's file location. It handles both regular Python files
	and package directories, ensuring reliable path resolution across different
	installation methods and environments.

	Returns:
		pathPackage: Absolute path to the package directory containing the module files.

	Notes:
		The function automatically handles the case where module introspection
		returns a file path by extracting the parent directory, ensuring the
		returned path always points to the package directory itself.
	"""
	pathPackage: Path = Path(inspect_getfile(importlib_import_module(packageNamePACKAGING)))
	if pathPackage.is_file():
		pathPackage = pathPackage.parent
	return pathPackage

@dataclasses.dataclass
class PackageSettings:
	"""
	Centralized configuration container for all package-wide settings.

	This dataclass serves as the single source of truth for package configuration,
	providing both static and dynamically-resolved values needed throughout the
	package lifecycle. The metadata on each field indicates when that value is
	determined - either during packaging or at installation/runtime.

	The design supports different evaluation phases to optimize performance and
	reliability:
	- Packaging-time: Values that can be determined during package creation
	- Installing-time: Values that require filesystem access or module introspection

	Attributes:
		fileExtension: Standard file extension for Python modules in this package.
		packageName: Canonical name of the package as defined in project configuration.
		pathPackage: Absolute filesystem path to the installed package directory.
		concurrencyPackage: Package to use for concurrent execution operations.
	"""
	fileExtension: str = dataclasses.field(default='.py', metadata={'evaluateWhen': 'installing'})
	packageName: str = dataclasses.field(default = packageNamePACKAGING, metadata={'evaluateWhen': 'packaging'})
	pathPackage: Path = dataclasses.field(default_factory=getPathPackageINSTALLING, metadata={'evaluateWhen': 'installing'})
	concurrencyPackage: str | None = None
	"""
	Package identifier for concurrent execution operations.

	Specifies which Python package should be used for parallel processing
	in computationally intensive operations. When None, the default concurrency
	package specified in the module constants is used. Accepted values include
	'multiprocessing' for standard parallel processing and 'numba' for
	specialized numerical computations.
	"""

concurrencyPackage = concurrencyPackageHARDCODED
"""
Active concurrency package configuration for the current session.

This module-level variable holds the currently selected concurrency package
identifier, initialized from the hardcoded default but available for runtime
modification through the package settings system.
"""

packageSettings = PackageSettings(concurrencyPackage=concurrencyPackage)
"""
Global package settings instance providing access to all configuration values.

This singleton instance serves as the primary interface for accessing package
configuration throughout the codebase. It combines statically-defined defaults
with dynamically-resolved values to provide a complete configuration profile
for the current package installation and runtime environment.
"""
