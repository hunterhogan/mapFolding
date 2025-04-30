from collections.abc import Callable
from importlib import import_module as importlib_import_module
from mapFolding import ComputationState, PackageSettings
from types import ModuleType
import dataclasses

@dataclasses.dataclass
class PackageInformation(PackageSettings):
	"""
	In _all_ of the root directory and "syntheticModules", the _only_ `PackageInformation` that are used are:
		The.concurrencyPackage
		The.dispatcher
	"""

	logicalPathModuleDispatcher: str | None = None
	"""Logical import path to the module containing the dispatcher function."""

	callableDispatcher: str | None = None
	"""Name of the function within the dispatcher module that will be called."""

	concurrencyPackage: str | None = None
	"""Package to use for concurrent execution (e.g., 'multiprocessing', 'numba')."""

	# "Evaluate When Packaging" and "Evaluate When Installing" https://github.com/hunterhogan/mapFolding/issues/18
	dataclassIdentifier: str = dataclasses.field(default='ComputationState', metadata={'evaluateWhen': 'packaging'})
	"""Name of the dataclass used to track computation state."""

	dataclassInstance: str = dataclasses.field(default='state', metadata={'evaluateWhen': 'packaging'})
	"""Default variable name for instances of the computation state dataclass."""

	dataclassInstanceTaskDistributionSuffix: str = dataclasses.field(default='Parallel', metadata={'evaluateWhen': 'packaging'})
	"""Suffix added to dataclassInstance for parallel task distribution."""

	dataclassModule: str = dataclasses.field(default='beDRY', metadata={'evaluateWhen': 'packaging'})
	"""Module containing the computation state dataclass definition."""

	datatypePackage: str = dataclasses.field(default='numpy', metadata={'evaluateWhen': 'packaging'})
	"""Package providing the numeric data types used in computation."""

	sourceAlgorithm: str = dataclasses.field(default='theDao', metadata={'evaluateWhen': 'packaging'})
	"""Module containing the reference implementation of the algorithm."""

	sourceCallableDispatcher: str = dataclasses.field(default='doTheNeedful', metadata={'evaluateWhen': 'packaging'})
	"""Name of the function that dispatches computation in the source algorithm."""

	sourceCallableInitialize: str = dataclasses.field(default='countInitialize', metadata={'evaluateWhen': 'packaging'})
	"""Name of the function that initializes computation in the source algorithm."""

	sourceCallableParallel: str = dataclasses.field(default='countParallel', metadata={'evaluateWhen': 'packaging'})
	"""Name of the function that performs parallel computation in the source algorithm."""

	sourceCallableSequential: str = dataclasses.field(default='countSequential', metadata={'evaluateWhen': 'packaging'})
	"""Name of the function that performs sequential computation in the source algorithm."""

	sourceConcurrencyManagerIdentifier: str = dataclasses.field(default='submit', metadata={'evaluateWhen': 'packaging'})
	"""Method name used to submit tasks to the concurrency manager."""

	sourceConcurrencyManagerNamespace: str = dataclasses.field(default='concurrencyManager', metadata={'evaluateWhen': 'packaging'})
	"""Variable name used for the concurrency manager instance."""

	sourceConcurrencyPackage: str = dataclasses.field(default='multiprocessing', metadata={'evaluateWhen': 'packaging'})
	"""Default package used for concurrency in the source algorithm."""

	dataclassInstanceTaskDistribution: str = dataclasses.field(default=None, metadata={'evaluateWhen': 'packaging'}) # pyright: ignore[reportAssignmentType]
	"""Variable name for the parallel distribution instance of the computation state."""

	logicalPathModuleDataclass: str = dataclasses.field(default=None, metadata={'evaluateWhen': 'packaging'}) # pyright: ignore[reportAssignmentType]
	"""Fully qualified import path to the module containing the computation state dataclass."""

	logicalPathModuleSourceAlgorithm: str = dataclasses.field(default=None, metadata={'evaluateWhen': 'packaging'}) # pyright: ignore[reportAssignmentType]
	"""Fully qualified import path to the module containing the source algorithm."""

	@property
	def dispatcher(self) -> Callable[['ComputationState'], 'ComputationState']:
		""" _The_ callable that connects `countFolds` to the logic that does the work."""
		logicalPath: str = self.logicalPathModuleDispatcher or self.logicalPathModuleSourceAlgorithm
		identifier: str = self.callableDispatcher or self.sourceCallableDispatcher
		moduleImported: ModuleType = importlib_import_module(logicalPath)
		return getattr(moduleImported, identifier)

	def __post_init__(self) -> None:
		if self.dataclassInstanceTaskDistribution is None: # pyright: ignore[reportUnnecessaryComparison]
			self.dataclassInstanceTaskDistribution = self.dataclassInstance + self.dataclassInstanceTaskDistributionSuffix

		if self.logicalPathModuleDataclass is None: # pyright: ignore[reportUnnecessaryComparison]
			self.logicalPathModuleDataclass = '.'.join([self.packageName, self.dataclassModule])
		if self.logicalPathModuleSourceAlgorithm is None: # pyright: ignore[reportUnnecessaryComparison]
			self.logicalPathModuleSourceAlgorithm = '.'.join([self.packageName, self.sourceAlgorithm])

class raiseIfNoneGitHubIssueNumber3(Exception): pass

logicalPathModuleDispatcherHARDCODED: str = 'mapFolding.syntheticModules.numbaCount'
callableDispatcherHARDCODED: str = 'doTheNeedful'
concurrencyPackageHARDCODED = 'multiprocessing'

The = PackageInformation(logicalPathModuleDispatcher=logicalPathModuleDispatcherHARDCODED, callableDispatcher=callableDispatcherHARDCODED, concurrencyPackage=concurrencyPackageHARDCODED)
