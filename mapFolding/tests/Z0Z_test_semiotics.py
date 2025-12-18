"""Tests for mapFolding/_semiotics.py and mapFolding/_e/_semiotics.py."""

from mapFolding._e._semiotics import (
	leafOrigin, pileOrigin, 一, 七, 三, 九, 二, 五, 八, 六, 四, 零, 首一, 首一三, 首一二, 首一二三, 首三, 首二, 首二三, 首零, 首零一, 首零一三, 首零一二, 首零一二三, 首零三,
	首零二, 首零二三)
from mapFolding._semiotics import decreasing, inclusive
from mapFolding.tests.dataSamples.semioticsData import (
	expectedDecreasing, expectedInclusive, expectedLeafOrigin, expectedPileOrigin, expected一, expected七, expected三,
	expected九, expected二, expected五, expected八, expected六, expected四, expected零, expected首一, expected首一三, expected首一二,
	expected首一二三, expected首三, expected首二, expected首二三, expected首零, expected首零一, expected首零一三, expected首零一二, expected首零一二三,
	expected首零三, expected首零二, expected首零二三)
import pytest

class TestSemioticsConstants:
	"""Tests for constant values in mapFolding/_semiotics.py."""

	@pytest.mark.parametrize("actual, expected", [
		(decreasing, expectedDecreasing),
		(inclusive, expectedInclusive),
	])
	def test_semanticReplacements(self, actual: int, expected: int) -> None:
		assert actual == expected, f"Expected {expected}, got {actual}"


class TestDimensionIndexConstants:
	"""Tests for dimension index constants (powers of 2) in mapFolding/_e/_semiotics.py."""

	@pytest.mark.parametrize("actual, expected, identifier", [
		(零, expected零, "零"),
		(一, expected一, "一"),
		(二, expected二, "二"),
		(三, expected三, "三"),
		(四, expected四, "四"),
		(五, expected五, "五"),
		(六, expected六, "六"),
		(七, expected七, "七"),
		(八, expected八, "八"),
		(九, expected九, "九"),
	])
	def test_dimensionIndexPowersOfTwo(self, actual: int, expected: int, identifier: str) -> None:
		assert actual == expected, f"Expected {identifier} = {expected}, got {actual}"


class TestOriginConstants:
	"""Tests for origin constants in mapFolding/_e/_semiotics.py."""

	@pytest.mark.parametrize("actual, expected, identifier", [
		(leafOrigin, expectedLeafOrigin, "leafOrigin"),
		(pileOrigin, expectedPileOrigin, "pileOrigin"),
	])
	def test_originConstants(self, actual: int, expected: int, identifier: str) -> None:
		assert actual == expected, f"Expected {identifier} = {expected}, got {actual}"


class TestDimensionCoordinateFunctions:
	"""Tests for cached functions that encode dimension coordinates."""

	@pytest.mark.parametrize("dimensionsTotal, expected", expected首零)
	def test_首零(self, dimensionsTotal: int, expected: int) -> None:
		actual = 首零(dimensionsTotal)
		assert actual == expected, f"首零({dimensionsTotal}): expected {expected}, got {actual}"

	@pytest.mark.parametrize("dimensionsTotal, expected", expected首零一)
	def test_首零一(self, dimensionsTotal: int, expected: int) -> None:
		actual = 首零一(dimensionsTotal)
		assert actual == expected, f"首零一({dimensionsTotal}): expected {expected}, got {actual}"

	@pytest.mark.parametrize("dimensionsTotal, expected", expected首零一二)
	def test_首零一二(self, dimensionsTotal: int, expected: int) -> None:
		actual = 首零一二(dimensionsTotal)
		assert actual == expected, f"首零一二({dimensionsTotal}): expected {expected}, got {actual}"

	@pytest.mark.parametrize("dimensionsTotal, expected", expected首零二)
	def test_首零二(self, dimensionsTotal: int, expected: int) -> None:
		actual = 首零二(dimensionsTotal)
		assert actual == expected, f"首零二({dimensionsTotal}): expected {expected}, got {actual}"

	@pytest.mark.parametrize("dimensionsTotal, expected", expected首一)
	def test_首一(self, dimensionsTotal: int, expected: int) -> None:
		actual = 首一(dimensionsTotal)
		assert actual == expected, f"首一({dimensionsTotal}): expected {expected}, got {actual}"

	@pytest.mark.parametrize("dimensionsTotal, expected", expected首一二)
	def test_首一二(self, dimensionsTotal: int, expected: int) -> None:
		actual = 首一二(dimensionsTotal)
		assert actual == expected, f"首一二({dimensionsTotal}): expected {expected}, got {actual}"

	@pytest.mark.parametrize("dimensionsTotal, expected", expected首二)
	def test_首二(self, dimensionsTotal: int, expected: int) -> None:
		actual = 首二(dimensionsTotal)
		assert actual == expected, f"首二({dimensionsTotal}): expected {expected}, got {actual}"

	@pytest.mark.parametrize("dimensionsTotal, expected", expected首三)
	def test_首三(self, dimensionsTotal: int, expected: int) -> None:
		actual = 首三(dimensionsTotal)
		assert actual == expected, f"首三({dimensionsTotal}): expected {expected}, got {actual}"

	@pytest.mark.parametrize("dimensionsTotal, expected", expected首零一二三)
	def test_首零一二三(self, dimensionsTotal: int, expected: int) -> None:
		actual = 首零一二三(dimensionsTotal)
		assert actual == expected, f"首零一二三({dimensionsTotal}): expected {expected}, got {actual}"

	@pytest.mark.parametrize("dimensionsTotal, expected", expected首零一三)
	def test_首零一三(self, dimensionsTotal: int, expected: int) -> None:
		actual = 首零一三(dimensionsTotal)
		assert actual == expected, f"首零一三({dimensionsTotal}): expected {expected}, got {actual}"

	@pytest.mark.parametrize("dimensionsTotal, expected", expected首零二三)
	def test_首零二三(self, dimensionsTotal: int, expected: int) -> None:
		actual = 首零二三(dimensionsTotal)
		assert actual == expected, f"首零二三({dimensionsTotal}): expected {expected}, got {actual}"

	@pytest.mark.parametrize("dimensionsTotal, expected", expected首零三)
	def test_首零三(self, dimensionsTotal: int, expected: int) -> None:
		actual = 首零三(dimensionsTotal)
		assert actual == expected, f"首零三({dimensionsTotal}): expected {expected}, got {actual}"

	@pytest.mark.parametrize("dimensionsTotal, expected", expected首一二三)
	def test_首一二三(self, dimensionsTotal: int, expected: int) -> None:
		actual = 首一二三(dimensionsTotal)
		assert actual == expected, f"首一二三({dimensionsTotal}): expected {expected}, got {actual}"

	@pytest.mark.parametrize("dimensionsTotal, expected", expected首一三)
	def test_首一三(self, dimensionsTotal: int, expected: int) -> None:
		actual = 首一三(dimensionsTotal)
		assert actual == expected, f"首一三({dimensionsTotal}): expected {expected}, got {actual}"

	@pytest.mark.parametrize("dimensionsTotal, expected", expected首二三)
	def test_首二三(self, dimensionsTotal: int, expected: int) -> None:
		actual = 首二三(dimensionsTotal)
		assert actual == expected, f"首二三({dimensionsTotal}): expected {expected}, got {actual}"
