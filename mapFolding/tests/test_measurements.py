"""Tests for mapFolding._e._measure module.

Tests verify each function against pre-computed verification data stored in
dataSamples/measurementData.py. The verification data was independently validated
using pure Python implementations without gmpy2 dependencies.

Each function is tested across its valid input range (0-256 inclusive for most,
2-256 for leafInSubHyperplane, 5-256 for ptount) and also tested for proper
exception handling on invalid inputs.
"""

from mapFolding._e import (
	dimensionNearestTail, dimensionNearest首, dimensionSecondNearest首, dimensionThirdNearest首,
	howManyDimensionsHaveOddParity, leafInSubHyperplane, ptount)
from mapFolding.tests.conftest import standardizedEqualToCallableReturn
from mapFolding.tests.dataSamples.measurementData import (
	dataDimensionNearest, dataDimensionSecondNearest, dataDimensionThirdNearest, dataHowMany0coordinatesAtTail,
	dataHowManyDimensionsHaveOddParity, dataLeafInSubHyperplane, dataPtount)
import pytest

@pytest.fixture
def dataMeasurementDimensionNearest() -> dict[int, int]:
	"""Provide verification data for dimensionNearest首."""
	return dataDimensionNearest

@pytest.fixture
def dataMeasurementDimensionSecondNearest() -> dict[int, int | None]:
	"""Provide verification data for dimensionSecondNearest首."""
	return dataDimensionSecondNearest

@pytest.fixture
def dataMeasurementDimensionThirdNearest() -> dict[int, int | None]:
	"""Provide verification data for dimensionThirdNearest首."""
	return dataDimensionThirdNearest

@pytest.fixture
def dataMeasurementLeafInSubHyperplane() -> dict[int, int]:
	"""Provide verification data for leafInSubHyperplane."""
	return dataLeafInSubHyperplane

@pytest.fixture
def dataMeasurementHowMany0coordinatesAtTail() -> dict[int, int]:
	"""Provide verification data for howMany0coordinatesAtTail."""
	return dataHowMany0coordinatesAtTail

@pytest.fixture
def dataMeasurementHowManyDimensionsHaveOddParity() -> dict[int, int]:
	"""Provide verification data for howManyDimensionsHaveOddParity."""
	return dataHowManyDimensionsHaveOddParity

@pytest.fixture
def dataMeasurementPtount() -> dict[int, int]:
	"""Provide verification data for ptount."""
	return dataPtount

class TestDimensionNearest:
	"""Tests for dimensionNearest首: finds 0-indexed position of MSB."""

	@pytest.mark.parametrize(
		"inputValue,expectedResult",
		[
			(inputValue, expectedResult)
			for inputValue, expectedResult in dataDimensionNearest.items()
		],
		ids=[f"n={inputValue}" for inputValue in dataDimensionNearest],
	)
	def test_dimensionNearest_validInputs(
		self,
		inputValue: int,
		expectedResult: int,
	) -> None:
		standardizedEqualToCallableReturn(expectedResult, dimensionNearest首, inputValue)

	@pytest.mark.parametrize(
		"invalidInput",
		[-1, -7, -13, -256],
		ids=["n=-1", "n=-7", "n=-13", "n=-256"],
	)
	def test_dimensionNearest_negativeInputsRaiseValueError(
		self,
		invalidInput: int,
	) -> None:
		standardizedEqualToCallableReturn(ValueError, dimensionNearest首, invalidInput)

class TestDimensionSecondNearest:
	"""Tests for dimensionSecondNearest首: finds 0-indexed position of second MSB."""

	@pytest.mark.parametrize(
		"inputValue,expectedResult",
		[
			(inputValue, expectedResult)
			for inputValue, expectedResult in dataDimensionSecondNearest.items()
		],
		ids=[f"n={inputValue}" for inputValue in dataDimensionSecondNearest],
	)
	def test_dimensionSecondNearest_validInputs(
		self,
		inputValue: int,
		expectedResult: int | None,
	) -> None:
		standardizedEqualToCallableReturn(expectedResult, dimensionSecondNearest首, inputValue)

	@pytest.mark.parametrize(
		"invalidInput",
		[-1, -5, -11, -128],
		ids=["n=-1", "n=-5", "n=-11", "n=-128"],
	)
	def test_dimensionSecondNearest_negativeInputsRaiseValueError(
		self,
		invalidInput: int,
	) -> None:
		standardizedEqualToCallableReturn(ValueError, dimensionSecondNearest首, invalidInput)

class TestDimensionThirdNearest:
	"""Tests for dimensionThirdNearest首: finds 0-indexed position of third MSB."""

	@pytest.mark.parametrize(
		"inputValue,expectedResult",
		[
			(inputValue, expectedResult)
			for inputValue, expectedResult in dataDimensionThirdNearest.items()
		],
		ids=[f"n={inputValue}" for inputValue in dataDimensionThirdNearest],
	)
	def test_dimensionThirdNearest_validInputs(
		self,
		inputValue: int,
		expectedResult: int | None,
	) -> None:
		standardizedEqualToCallableReturn(expectedResult, dimensionThirdNearest首, inputValue)

	@pytest.mark.parametrize(
		"invalidInput",
		[-1, -3, -17, -64],
		ids=["n=-1", "n=-3", "n=-17", "n=-64"],
	)
	def test_dimensionThirdNearest_negativeInputsRaiseValueError(
		self,
		invalidInput: int,
	) -> None:
		standardizedEqualToCallableReturn(ValueError, dimensionThirdNearest首, invalidInput)

class TestLeafInSubHyperplane:
	"""Tests for leafInSubHyperplane: projects hyperplane leaf onto sub-hyperplane."""

	@pytest.mark.parametrize(
		"inputValue,expectedResult",
		[
			(inputValue, expectedResult)
			for inputValue, expectedResult in dataLeafInSubHyperplane.items()
		],
		ids=[f"n={inputValue}" for inputValue in dataLeafInSubHyperplane],
	)
	def test_leafInSubHyperplane_validInputs(
		self,
		inputValue: int,
		expectedResult: int,
	) -> None:
		standardizedEqualToCallableReturn(expectedResult, leafInSubHyperplane, inputValue)

	@pytest.mark.parametrize(
		"invalidInput",
		[0, -1, -7, -19],
		ids=["n=0", "n=-1", "n=-7", "n=-19"],
	)
	def test_leafInSubHyperplane_invalidInputsRaiseValueError(
		self,
		invalidInput: int,
	) -> None:
		standardizedEqualToCallableReturn(ValueError, leafInSubHyperplane, invalidInput)

class TestHowMany0coordinatesAtTail:
	"""Tests for howMany0coordinatesAtTail: counts trailing zeros (CTZ)."""

	@pytest.mark.parametrize(
		"inputValue,expectedResult",
		[
			(inputValue, expectedResult)
			for inputValue, expectedResult in dataHowMany0coordinatesAtTail.items()
		],
		ids=[f"n={inputValue}" for inputValue in dataHowMany0coordinatesAtTail],
	)
	def test_howMany0coordinatesAtTail_validInputs(
		self,
		inputValue: int,
		expectedResult: int,
	) -> None:
		standardizedEqualToCallableReturn(expectedResult, dimensionNearestTail, inputValue)

	@pytest.mark.parametrize(
		"invalidInput",
		[-1, -2, -8, -37],
		ids=["n=-1", "n=-2", "n=-8", "n=-37"],
	)
	def test_howMany0coordinatesAtTail_negativeInputsRaiseValueError(
		self,
		invalidInput: int,
	) -> None:
		standardizedEqualToCallableReturn(ValueError, dimensionNearestTail, invalidInput)

class TestHowManyDimensionsHaveOddParity:
	"""Tests for howManyDimensionsHaveOddParity: bit_count - 1, minimum 0."""

	@pytest.mark.parametrize(
		"inputValue,expectedResult",
		[
			(inputValue, expectedResult)
			for inputValue, expectedResult in dataHowManyDimensionsHaveOddParity.items()
		],
		ids=[f"n={inputValue}" for inputValue in dataHowManyDimensionsHaveOddParity],
	)
	def test_howManyDimensionsHaveOddParity_validInputs(
		self,
		inputValue: int,
		expectedResult: int,
	) -> None:
		standardizedEqualToCallableReturn(expectedResult, howManyDimensionsHaveOddParity, inputValue)

	@pytest.mark.parametrize(
		"invalidInput",
		[-1, -5, -23, -89],
		ids=["n=-1", "n=-5", "n=-23", "n=-89"],
	)
	def test_howManyDimensionsHaveOddParity_negativeInputsRaiseValueError(
		self,
		invalidInput: int,
	) -> None:
		standardizedEqualToCallableReturn(ValueError, howManyDimensionsHaveOddParity, invalidInput)

class TestPtount:
	"""Tests for ptount: measures distance from power of two's bit count after subtracting 3."""

	@pytest.mark.parametrize(
		"inputValue,expectedResult",
		[
			(inputValue, expectedResult)
			for inputValue, expectedResult in dataPtount.items()
		],
		ids=[f"n={inputValue}" for inputValue in dataPtount],
	)
	def test_ptount_validInputs(
		self,
		inputValue: int,
		expectedResult: int,
	) -> None:
		standardizedEqualToCallableReturn(expectedResult, ptount, inputValue)

	@pytest.mark.parametrize(
		"invalidInput",
		[0, 1, 2, -1, -7, -41],
		ids=["n=0", "n=1", "n=2", "n=-1", "n=-7", "n=-41"],
	)
	def test_ptount_invalidInputsRaiseValueError(
		self,
		invalidInput: int,
	) -> None:
		standardizedEqualToCallableReturn(ValueError, ptount, invalidInput)
