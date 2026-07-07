"""Test framework infrastructure and shared fixtures for mapFolding.

This module serves as the foundation for the entire test suite, providing standardized
fixtures, temporary file management, and testing utilities. It implements the Single
Source of Truth principle for test configuration and establishes consistent patterns
that make the codebase easier to extend and maintain.

The testing framework is designed for multiple audiences:
- Contributors who need to understand the test patterns
- AI assistants that help maintain or extend the codebase
- Users who want to test custom modules they create
- Future maintainers who need to debug or modify tests

Key Components:
- Temporary file management with automatic cleanup
- Standardized assertion functions with uniform error messages
- Test data generation from OEIS sequences for reproducible results
- Mock objects for external dependencies and timing-sensitive operations

The module follows Domain-Driven Design principles, organizing test concerns around
the mathematical concepts of map folding rather than technical implementation details.
This makes tests more meaningful and easier to understand in the context of the
research domain.
"""

from __future__ import annotations

from mapFolding import _theSSOT, packageSettings
from mapFolding.oeis import oeisIDsImplemented
from pathlib import Path
from typing import TYPE_CHECKING
import pickle
import pytest
import random
import shutil
import uuid
import warnings

if TYPE_CHECKING:
	from collections.abc import Callable, Generator
	from numpy.typing import NDArray
	from pytest import FixtureRequest
	from typing import Any
	import numpy

# ================== Test-function parameters ======================================================

@pytest.fixture()
def approx_abs(request: FixtureRequest) -> float:
	"""The `abs` (***abs***olute tolerance) parameter value for `pytest.approx`."""
	return 1e-12

@pytest.fixture()
def approx_rel(request: FixtureRequest) -> float:
	"""The `rel` (***rel***ative tolerance) parameter value for `pytest.approx`."""
	return 1e-6

@pytest.fixture()
def atol(request: FixtureRequest) -> float:
	"""The `atol` (***a***bsolute ***tol***erance) parameter value for `numpy.allclose`."""
	return 1e-08

@pytest.fixture()
def rtol(request: FixtureRequest) -> float:
	"""The `rtol` (***r***elative ***tol***erance) parameter value for `numpy.allclose`."""
	return 1e-05

@pytest.fixture(autouse=True)
def setupWarningsAsErrors() -> Generator[None, Any]:
	"""Convert all warnings to errors for all tests."""
	warnings.filterwarnings("error")
	yield
	warnings.resetwarnings()

#======== SSOT for test data paths and filenames ==============
# TODO I might still need something like
# this to test the creation of a job. But I don't need to use this for every tmp dir or file, and it
# doesn't need to be this complicated.
pathDataSamples: Path = Path(packageSettings.pathPackage, "tests/dataSamples").absolute()
path_tmpRoot: Path = pathDataSamples / "tmp"
path_tmpRoot.mkdir(parents=True, exist_ok=True)

# The registrar maintains the register of tmp filesystem objects
registerOfTemporaryFilesystemObjects: set[Path] = set()

def registrarDeletesTemporaryFilesystemObjects() -> None:
	"""The registrar cleans up tmp filesystem objects in the register."""
	for path_tmp in sorted(registerOfTemporaryFilesystemObjects, reverse=True):
		if path_tmp.is_file():
			path_tmp.unlink(missing_ok=True)
		elif path_tmp.is_dir():
			shutil.rmtree(path_tmp, ignore_errors=True)
	registerOfTemporaryFilesystemObjects.clear()

def registrarRecordsTemporaryFilesystemObject(path: Path) -> None:
	"""The registrar adds a tmp filesystem object to the register.

	Parameters
	----------
	path : Path
		The filesystem path to register for cleanup.

	"""
	registerOfTemporaryFilesystemObjects.add(path)

@pytest.fixture
def pathCacheTesting(path_tmpTesting: Path) -> Generator[Path, Any]:
	"""Temporarily replace the OEIS cache directory with a test directory.

	Parameters
	----------
	path_tmpTesting : Path
		Temporary directory path from the `path_tmpTesting` fixture.

	Yields
	------
	temporaryCachePath : Generator[Path, Any, None]
		Context manager that provides the temporary cache path and restores original.

	"""
	pathCacheOriginal: Path = _theSSOT.pathCache
	_theSSOT.pathCache = path_tmpTesting
	yield path_tmpTesting
	_theSSOT.pathCache = pathCacheOriginal

@pytest.fixture
def pathFilenameFoldsTotalTesting(path_tmpTesting: Path) -> Path:
	"""Creates a temporary file path for folds total testing.

	Parameters
	----------
	path_tmpTesting : Path
		Temporary directory path from the `path_tmpTesting` fixture.

	Returns
	-------
	foldsTotalFilePath : Path
		Path to a temporary file for testing folds total functionality.

	"""
	return path_tmpTesting.joinpath("foldsTotalTest.txt")

@pytest.fixture
def pathFilename_tmpTesting(request: pytest.FixtureRequest) -> Path:
	"""Creates a unique temporary file path for testing.

	Parameters
	----------
	request : pytest.FixtureRequest
		The pytest request object, optionally containing `param` for file extension.

	Returns
	-------
	temporaryFilePath : Path
		Path to a unique temporary file that will be cleaned up automatically.

	"""
	try:
		extension = request.param
	except AttributeError:
		extension = ".txt"

	# "Z0Z_" ensures the name does not start with a number, which would make it an invalid Python identifier
	uuid_hex: str = uuid.uuid4().hex
	relativePath: str = "Z0Z_" + uuid_hex[0:-8]
	filenameStem: str = "Z0Z_" + uuid_hex[-8:None]

	pathFilename_tmp = Path(path_tmpRoot, relativePath, filenameStem + extension)
	pathFilename_tmp.parent.mkdir(parents=True, exist_ok=False)

	registrarRecordsTemporaryFilesystemObject(pathFilename_tmp.parent)
	return pathFilename_tmp

@pytest.fixture
def path_tmpTesting(request: pytest.FixtureRequest) -> Path:
	"""Creates a unique temporary directory for testing.

	Parameters
	----------
	request : pytest.FixtureRequest
		The pytest request object providing test context.

	Returns
	-------
	temporaryPath : Path
		Path to a unique temporary directory that will be cleaned up automatically.

	"""
	# "Z0Z_" ensures the directory name does not start with a number, which would make it an invalid Python identifier
	uuid_hex: str = uuid.uuid4().hex
	path_tmp: Path = path_tmpRoot / ("Z0Z_" + uuid_hex)
	path_tmp.mkdir(parents=True, exist_ok=False)

	registrarRecordsTemporaryFilesystemObject(path_tmp)
	return path_tmp

@pytest.fixture(scope="session", autouse=True)
def setupTeardownTemporaryFilesystemObjects() -> Generator[None]:
	"""Auto-fixture to setup test data directories and cleanup after."""
	pathDataSamples.mkdir(exist_ok=True)
	path_tmpRoot.mkdir(exist_ok=True)
	yield
	registrarDeletesTemporaryFilesystemObjects()

#======== OEIS ids =====================================

@pytest.fixture(params=oeisIDsImplemented)
def oeisIDmapFolding(request: pytest.FixtureRequest) -> Any:
	"""Parametrized fixture providing all implemented OEIS sequence identifiers.

	Parameters
	----------
	request : pytest.FixtureRequest
		The pytest request object containing the current parameter value.

	Returns
	-------
	sequenceIdentifier : Any
		OEIS sequence identifier for testing across all implemented sequences.
	"""
	return request.param

@pytest.fixture
def oeisID_1random() -> str:
	"""Return one random valid OEIS ID.

	Returns
	-------
	randomSequenceIdentifier : str
		Randomly selected OEIS sequence identifier from implemented sequences.
	"""
	return random.choice(oeisIDsImplemented)

#======== Miscellaneous =====================================

@pytest.fixture
def loadArrayFoldings() -> Callable[[int], NDArray[numpy.uint8]]:
	"""Factory fixture for loading pickled array foldings data.

	Returns
	-------
	loaderFunction : Callable[[int], NDArray[numpy.uint8]]
		Function that loads arrayFoldings for a given dimensionsTotal.
	"""
	def loader(dimensionsTotal: int) -> NDArray[numpy.uint8]:
		pathFilename = pathDataSamples / f"arrayFoldingsP2d{dimensionsTotal}.pkl"
		arrayFoldings: NDArray[numpy.uint8] = pickle.loads(pathFilename.read_bytes())  # noqa: S301
		return arrayFoldings

	return loader
