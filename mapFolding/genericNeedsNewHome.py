# ruff: noqa: DOC201, D100, D103
from __future__ import annotations

from collections.abc import Sequence
from gmpy2 import sign
from humpy_cytoolz import curry as syntacticCurry
from hunterMakesPy.parseParameters import intInnit
from more_itertools import extract
from operator import eq
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Container, Iterable, Iterator
	from hunterMakesPy import Ordinals

#======== Boolean antecedents ================================================

@syntacticCurry
def between吗[小于: Ordinals](floor: 小于, ceiling: 小于, comparand: 小于) -> bool:
	"""Inclusive `floor <= comparand <= ceiling`."""
	return floor <= comparand <= ceiling

# NOTE `个` TypeVar exists to help ty with static type checking. See https://github.com/astral-sh/ty/issues/2799.
def consecutive吗[个: Sequence[int]](flatContainer: 个) -> bool:
	"""Are the integers in `flatContainer` consecutive, either ascending or descending?"""
	ImaListOfInt: list[int] = intInnit(flatContainer, 'flatContainer', Sequence[int])
	difference: int = ImaListOfInt[-1] - ImaListOfInt[0]
	direction: int = sign(difference)
	rr = range(ImaListOfInt[0], ImaListOfInt[-1] + direction, direction)
	return (abs(difference) == (len(ImaListOfInt) - 1)) and (all(map(eq, ImaListOfInt, rr)))

@syntacticCurry
def thisHasThat吗[个](this: Container[个], that: 个) -> bool:
	"""You can test whether `that` is present in `this`.

	You can use `thisHasThat` in an `if` statement, or you can pass `thisHasThat` as a
	predicate to a filtering utility described in the module docstring.

	Parameters
	----------
	this : Container[个]
		Container to search.
	that : 个
		Value to find.

	Returns
	-------
	thatIsPresent : bool
		`True` if `that in this`.

	References
	----------
	[1] `operator.contains` (Python documentation)
		https://docs.python.org/3/library/operator.html#operator.contains

	"""
	return that in this

@syntacticCurry
def thisNotHaveThat吗[个](this: Container[个], that: 个) -> bool:
	return not thisHasThat吗(this, that)

#======== Filtering functions ================================================

def exclude[个](flatContainer: Sequence[个], indices: Iterable[int]) -> Iterator[个]:
	"""Yield items from `flatContainer` whose positions are not in `indices`."""
	lengthIterable: int = len(flatContainer)

	def normalizeIndex(index: int) -> int:
		if index < 0:
			index = (index + lengthIterable) % lengthIterable
		return index
	indicesInclude: list[int] = sorted(set(range(lengthIterable)).difference(map(normalizeIndex, indices)))
	return extract(flatContainer, indicesInclude)
