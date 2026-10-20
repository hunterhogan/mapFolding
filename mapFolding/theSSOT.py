"""Access shared package settings and bundled sample-data paths.

(AI generated docstring)

You can use this module to inspect package metadata and the default concurrency backend through
a shared settings instance.

Contents
--------
Classes
	mapFoldingPackageSettings
		Extend package settings with concurrency and sample-data locations.
Shared settings
	settingsPackage
		Access the initialized settings shared by package modules.
"""
from __future__ import annotations

from hunterMakesPy import PackageSettings
from pathlib import Path
import dataclasses

@dataclasses.dataclass
class mapFoldingPackageSettings(PackageSettings):
	"""Store package metadata, concurrency preferences, and sample-data locations.

	(AI generated docstring)

	You can use this class to extend `hunterMakesPy.PackageSettings` [1] with settings required by
	folding computations. Construction resolves package metadata through the inherited initializer
	and derives the bundled sample-data directory.

	Attributes
	----------
	identifierPackageFALLBACK : dataclasses.InitVar[str] = ''
		Constructor-only fallback package identifier; no instance field is retained for this value.
	pathPackage : Path
		Package directory resolved by the inherited initialization when the default path is supplied.
	identifierPackage : str = ''
		Package name resolved by the inherited initialization when a fallback is supplied.
	fileExtension : str = '.py'
		Default extension for package source files.
	concurrencyPackage : str = 'multiprocessing'
		Default backend label for processor-limit resolution.
	pathDataSamples : Path
		Derived directory `pathPackage / 'tests' / 'dataSamples'`. Initialization does not create
		the directory.

	Examples
	--------
	This module constructs the shared settings instance.

		```python
		settingsPackage = mapFoldingPackageSettings('mapFolding')
		```

	References
	----------
	[1] hunterMakesPy `PackageSettings`.
		https://context7.com/hunterhogan/huntermakespy
	"""

	concurrencyPackage: str = 'multiprocessing'
	"""Package identifier for concurrent execution operations."""

	pathDataSamples: Path = dataclasses.field(init=False, default=Path())
	"""Locate bundled sample data under `pathPackage / 'tests' / 'dataSamples'`.

	(AI generated docstring)
	"""

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
