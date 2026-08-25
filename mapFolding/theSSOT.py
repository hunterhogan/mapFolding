"""Access and configure package settings."""
from __future__ import annotations

from hunterMakesPy import PackageSettings
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

	concurrencyPackage : str = 'multiprocessing'
		Package identifier for concurrent execution operations.
	"""

	concurrencyPackage: str = 'multiprocessing'
	"""Package identifier for concurrent execution operations."""

	pathDataSamples: Path = dataclasses.field(init=False, default=Path())

	def __post_init__(self, identifierPackageFALLBACK: str) -> None:
		"""Finish package initialization from `identifierPackageFALLBACK` and derive `pathDataSamples`.

		(AI generated docstring)

		This method completes initialization of `mapFoldingPackageSettings`. This method delegates
		package discovery and package-root resolution to the inherited initialization logic using
		`identifierPackageFALLBACK`. This method then sets `pathDataSamples` to
		`pathPackage / 'tests' / 'dataSamples'` so the package can locate bundled sample data.

		Parameters
		----------
		identifierPackageFALLBACK : str
			Fallback package identifier used when automatic package discovery cannot resolve the
			package name during initialization.
		"""
		super().__post_init__(identifierPackageFALLBACK)
		self.pathDataSamples = self.pathPackage / 'tests' / 'dataSamples'

settingsPackage = mapFoldingPackageSettings('mapFolding')
"""Global package settings."""
