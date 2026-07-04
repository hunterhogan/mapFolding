from __future__ import annotations

from mapFolding import packageSettings
from pathlib import Path
from typing import TYPE_CHECKING
import pickle
import pytest

if TYPE_CHECKING:
	from collections.abc import Callable
	from numpy.typing import NDArray
	import numpy

pathDataSamples: Path = Path(packageSettings.pathPackage, "tests/dataSamples").absolute()

@pytest.fixture
def loadArrayFoldings() -> Callable[[int], NDArray[numpy.uint8]]:
	"""Factory fixture for loading pickled array foldings data.

	Returns
	-------
	loaderFunction : Callable[[int], NDArray[numpy.uint8]]
		Function that loads arrayFoldings for a given dimensionsTotal.
	"""
	def loader(dimensionsTotal: int) -> NDArray[numpy.uint8]:
		pathFilename: Path = pathDataSamples / f"arrayFoldingsP2d{dimensionsTotal}.pkl"
		arrayFoldings: NDArray[numpy.uint8] = pickle.loads(pathFilename.read_bytes())  # noqa: S301
		return arrayFoldings

	return loader

@pytest.fixture(params=(0.25,))
def CPUlimitPinningTests(request: pytest.FixtureRequest) -> float:
	return float(request.param)

@pytest.fixture(params=(2, 3, 4), ids=lambda pileDepth: f"pileDepth={pileDepth}")
def pileDepthPinningTests(request: pytest.FixtureRequest) -> int:
	return int(request.param)
