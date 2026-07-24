"""Access and configure package settings."""

from __future__ import annotations

from hunterMakesPy import PackageSettings
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

	concurrencyPackage : str = 'multiprocessing'
		Package identifier for concurrent execution operations.
	"""

	concurrencyPackage: str = 'multiprocessing'
	"""Package identifier for concurrent execution operations."""

identifierPackageFALLBACK = "mapFolding"
"""Manually entered package name used as fallback when dynamic resolution fails."""

settingsPackage = mapFoldingPackageSettings(
	identifierPackageFALLBACK=identifierPackageFALLBACK,
)
"""Global package settings."""
