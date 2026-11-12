"""The leaves connected to a leaf by a crease, in the documented order of the dimension that the crease crosses.

No four-dimensional recording of creases exists, so the maps start at five dimensions.
"""
from __future__ import annotations

from itertools import chain, repeat
from mapFolding.creases import getLeavesCreaseAnte, getLeavesCreasePost
from mapFolding.dataBaskets import StateElimination
from mapFolding.tests import assertEqualTo
from mapFolding.tests.dataSamples.maps2上nDimensional import (
	dictionaryCreasesDecreaseKnown, dictionaryCreasesIncreaseKnown, mapShape2上5Dimensional, mapShape2上6Dimensional)
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from collections.abc import Callable, Iterator
	from hunterMakesPy import CallableFunction

parametersGetLeavesCrease: dict[str, tuple[Callable[[StateElimination, int], Iterator[int]], tuple[int, ...], int, list[int]]] = dict(chain(
	zip(map('getLeavesCreasePost-mapShape2上5Dimensional-leaf{}'.format, dictionaryCreasesIncreaseKnown[mapShape2上5Dimensional])
		, zip(repeat(getLeavesCreasePost), repeat(mapShape2上5Dimensional), dictionaryCreasesIncreaseKnown[mapShape2上5Dimensional].keys()
			, dictionaryCreasesIncreaseKnown[mapShape2上5Dimensional].values()), strict=True)
	, zip(map('getLeavesCreasePost-mapShape2上6Dimensional-leaf{}'.format, dictionaryCreasesIncreaseKnown[mapShape2上6Dimensional])
		, zip(repeat(getLeavesCreasePost), repeat(mapShape2上6Dimensional), dictionaryCreasesIncreaseKnown[mapShape2上6Dimensional].keys()
			, dictionaryCreasesIncreaseKnown[mapShape2上6Dimensional].values()), strict=True)
	, zip(map('getLeavesCreaseAnte-mapShape2上5Dimensional-leaf{}'.format, dictionaryCreasesDecreaseKnown[mapShape2上5Dimensional])
		, zip(repeat(getLeavesCreaseAnte), repeat(mapShape2上5Dimensional), dictionaryCreasesDecreaseKnown[mapShape2上5Dimensional].keys()
			, dictionaryCreasesDecreaseKnown[mapShape2上5Dimensional].values()), strict=True)
	, zip(map('getLeavesCreaseAnte-mapShape2上6Dimensional-leaf{}'.format, dictionaryCreasesDecreaseKnown[mapShape2上6Dimensional])
		, zip(repeat(getLeavesCreaseAnte), repeat(mapShape2上6Dimensional), dictionaryCreasesDecreaseKnown[mapShape2上6Dimensional].keys()
			, dictionaryCreasesDecreaseKnown[mapShape2上6Dimensional].values()), strict=True)
))

@pytest.mark.parametrize('getLeavesCrease, mapShape, leaf, expected', parametersGetLeavesCrease.values(), ids=parametersGetLeavesCrease.keys())
def test_getLeavesCrease(getLeavesCrease: CallableFunction[[StateElimination, int], Iterator[int]], mapShape: tuple[int, ...], leaf: int
	, expected: list[int]) -> None:
	"""The leaves are the recorded leaves, in increasing order of the flipped dimension index, which is the order of `leaf ^ leafCrease`."""
	leavesCrease: tuple[int, ...] = tuple(getLeavesCrease(StateElimination(mapShape), leaf))
	assertEqualTo(sorted(leavesCrease), expected, getLeavesCrease.__name__, mapShape, leaf)
	assertEqualTo(leavesCrease, tuple(sorted(leavesCrease, key=leaf.__xor__)), getLeavesCrease.__name__, mapShape, leaf)
