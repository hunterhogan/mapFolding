from typing import Any, Generator, Set
import pathlib
import pytest
import shutil
import uuid

# SSOT for test data paths
pathDataSamples = pathlib.Path("tests/dataSamples")
pathTempRoot = pathDataSamples / "tmp"

# The registrar maintains the register of temp files
registerOfTempFiles: Set[pathlib.Path] = set()

def addTempFileToRegister(path: pathlib.Path) -> None:
    """The registrar adds a temp file to the register."""
    registerOfTempFiles.add(path)

def cleanupTempFileRegister() -> None:
    """The registrar cleans up temp files in the register."""
    for pathTemp in sorted(registerOfTempFiles, reverse=True):
        try:
            if pathTemp.is_file():
                pathTemp.unlink(missing_ok=True)
            elif pathTemp.is_dir():
                shutil.rmtree(pathTemp, ignore_errors=True)
        except Exception as ERRORmessage:
            print(f"Warning: Failed to clean up {pathTemp}: {ERRORmessage}")
    registerOfTempFiles.clear()

@pytest.fixture(scope="session", autouse=True)
def setupTeardownTestData() -> Generator[None, None, None]:
    """Auto-fixture to setup test data directories and cleanup after."""
    pathDataSamples.mkdir(exist_ok=True)
    pathTempRoot.mkdir(exist_ok=True)
    yield
    cleanupTempFileRegister()

@pytest.fixture
def pathTempTesting(request: pytest.FixtureRequest) -> pathlib.Path:
    """Create a unique temp directory for each test function."""
    # TODO I got rid of this shit. how the fuck is it back?
    # Sanitize test name for filesystem compatibility
    sanitizedName = request.node.name.replace('[', '_').replace(']', '_').replace('/', '_')
    uniqueDirectory = f"{sanitizedName}_{uuid.uuid4()}"
    pathTemp = pathTempRoot / uniqueDirectory
    pathTemp.mkdir(parents=True, exist_ok=True)

    addTempFileToRegister(pathTemp)
    return pathTemp

@pytest.fixture
def pathCacheTesting(pathTempTesting: pathlib.Path) -> Generator[pathlib.Path, Any, None]:
    """Temporarily replace the OEIS cache directory with a test directory."""
    from mapFolding import oeis as there_must_be_a_better_way
    pathCacheOriginal = there_must_be_a_better_way._pathCache
    there_must_be_a_better_way._pathCache = pathTempTesting
    yield pathTempTesting
    there_must_be_a_better_way._pathCache = pathCacheOriginal

@pytest.fixture
def pathFilenameFoldsTotalTesting(pathTempTesting: pathlib.Path) -> pathlib.Path:
    return pathTempTesting.joinpath("foldsTotalTest.txt")
