"""Count symmetric meander shapes by cutting the meander at the symmetry axis along the road.

(AI generated docstring)

You can use this module to compute A223096(m), the number of symmetric meander shapes on 2m + 1
crossings. The transfer is `mapFolding.algorithms.matrixMeanders.count`, unchanged, on the m piles of
one half. Each state is an `arcCode` of `mapFolding.algorithms.matrixMeanders`: the even bits,
`bitsAlfa`, describe the terminals on one side of the road, the odd bits, `bitsZulu`, describe the
terminals on the other side, and each side is read from the road outward with a sentinel bit at the
top. At the axis, `doTheNeedful` counts each half that joins with its half-turn image into one strand.

Contents
--------
Functions
	doTheNeedful
		Count the halves that join at the symmetry axis and add the count to `state.countAddend`.
	getBitsLocator
		Make the mask that selects the bits of one side of the road in an `arcCode`.
	isAxisMirrored吗
		Decide whether the symmetry axis can match the strands of the two sides of an `arcCode`.
	isHalfJoined吗
		Decide whether the half that an `arcCode` describes joins with its half-turn image.
	makeLookupPartner
		Pair each terminal on the cut with the other end of its arc, in circle order.
	pairTerminals
		Pair the terminals of one side that share an arc, and collect the crossing terminals.
	reflectArcCode
		Reflect an `arcCode` across the road.
"""
from __future__ import annotations

from mapFolding.algorithms.matrixAxisStrip import isJoined吗
from mapFolding.algorithms.matrixMeanders import count
from operator import add, mul
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterator
	from mapFolding.dataBaskets import StateMeandersAxis

# TODO remake `getBitsLocator` from scratch.
def getBitsLocator(arcCode: int) -> int:
	"""Make the mask that selects the bits of one side of the road in an `arcCode`.

	(AI generated docstring)

	Parameters
	----------
	arcCode : int
		State of the pile-cut transfer.

	Returns
	-------
	bitsLocator : int
		Mask `0b…010101` with a set bit at each even position up to the length of `arcCode`. `arcCode
		& bitsLocator` is `bitsAlfa`, and `arcCode >> 1 & bitsLocator` is `bitsZulu`.
	"""
	return (1 << (arcCode.bit_length() | 1) + 1) // 3

# TODO remake `reflectArcCode` from scratch.
def reflectArcCode(arcCode: int) -> int:
	"""Reflect an `arcCode` across the road.

	(AI generated docstring)

	The reflection σ exchanges `bitsAlfa` and `bitsZulu`.

	Parameters
	----------
	arcCode : int
		State of the pile-cut transfer.

	Returns
	-------
	arcCodeReflected : int
		State of the reflected half.
	"""
	bitsLocator: int = getBitsLocator(arcCode)
	return (arcCode & bitsLocator) << 1 | arcCode >> 1 & bitsLocator

# TODO remake `pairTerminals` from scratch.
def pairTerminals(bitsSide: int, boxOfTerminals: Iterator[int], lookupPartner: list[int]) -> list[int]:
	"""Pair the terminals of one side that share an arc, and collect the crossing terminals.

	(AI generated docstring)

	Each terminal of the side takes the next circle position from `boxOfTerminals`. A `0` bit is the
	inner end of an arc on the same side. A `1` bit is the outer end of that arc when an inner end is
	unmatched, and otherwise the end of an arc that crosses the road.

	Input Mutation
	--------------
	This function records each pair of the side in `lookupPartner`.

	Parameters
	----------
	bitsSide : int
		`bitsAlfa` or `bitsZulu` of an `arcCode`, read from the road outward in steps of two bits.
	boxOfTerminals : Iterator[int]
		Circle position of each terminal of the side, in the order of `bitsSide`.
	lookupPartner : list[int]
		Partner of each circle position.

	Returns
	-------
	boxOfCross : list[int]
		Circle positions of the terminals of the side whose arcs cross the road, in the order of
		`bitsSide`.
	"""
	boxOfOpen: list[int] = []
	boxOfCross: list[int] = []
	while 1 < bitsSide:
		terminal: int = next(boxOfTerminals)
		if bitsSide & 1 == 0:
			boxOfOpen.append(terminal)
		elif boxOfOpen:
			lookupPartner[terminal] = boxOfOpen.pop()
			lookupPartner[lookupPartner[terminal]] = terminal
		else:
			boxOfCross.append(terminal)
		bitsSide >>= 2
	return boxOfCross

# TODO remake `makeLookupPartner` from scratch.
def makeLookupPartner(arcCode: int) -> list[int]:
	"""Pair each terminal on the cut with the other end of its arc, in circle order.

	(AI generated docstring)

	The circle order lists the terminals of `bitsAlfa` from the road outward and then the terminals of
	`bitsZulu` from the outside to the road. The k-th crossing terminal of `bitsAlfa` pairs with the
	k-th crossing terminal of `bitsZulu`.

	Parameters
	----------
	arcCode : int
		State of the pile-cut transfer at the symmetry axis.

	Returns
	-------
	lookupPartner : list[int]
		List of N circle positions. `lookupPartner[terminal]` is the position of the terminal that
		shares an arc with `terminal`.
	"""
	bitsLocator: int = getBitsLocator(arcCode)
	bitsAlfa: int = arcCode & bitsLocator
	bitsZulu: int = arcCode >> 1 & bitsLocator
	lookupPartner: list[int] = list(range((bitsAlfa.bit_length() + bitsZulu.bit_length()) // 2 - 1))
	boxOfCrossAlfa: list[int] = pairTerminals(bitsAlfa, iter(range(len(lookupPartner))), lookupPartner)
	boxOfCrossZulu: list[int] = pairTerminals(bitsZulu, iter(range(len(lookupPartner) - 1, -1, -1)), lookupPartner)
	tuple(map(lookupPartner.__setitem__, boxOfCrossAlfa, boxOfCrossZulu, strict=True))
	tuple(map(lookupPartner.__setitem__, boxOfCrossZulu, boxOfCrossAlfa, strict=True))
	return lookupPartner

# TODO remake `isAxisMirrored吗` from scratch.
def isAxisMirrored吗(arcCode: int) -> bool:
	"""Decide whether the symmetry axis can match the strands of the two sides of an `arcCode`.

	(AI generated docstring)

	With L_alfa strands on the side of `bitsAlfa` and L_zulu strands on the side of `bitsZulu`, the
	symmetry axis can match the sides when L_alfa − L_zulu ∈ {0, 2}.

	Parameters
	----------
	arcCode : int
		State of the pile-cut transfer at the symmetry axis.

	Returns
	-------
	isAxisMirrored : bool
		`True` when L_alfa − L_zulu ∈ {0, 2}.
	"""
	bitsLocator: int = getBitsLocator(arcCode)
	return (arcCode >> 1 & bitsLocator).bit_length() <= (arcCode & bitsLocator).bit_length() <= (arcCode >> 1 & bitsLocator).bit_length() + 4

# TODO remake `isHalfJoined吗` from scratch.
def isHalfJoined吗(arcCode: int) -> bool:
	"""Decide whether the half that an `arcCode` describes joins with its half-turn image.

	(AI generated docstring)

	The test takes the center terminal on the side of `bitsAlfa`. The half joins when
	`isAxisMirrored吗` accepts the numbers of strands and
	`mapFolding.algorithms.matrixAxisStrip.isJoined吗` accepts the circle order.

	Parameters
	----------
	arcCode : int
		State of the pile-cut transfer at the symmetry axis.

	Returns
	-------
	isHalfJoined : bool
		`True` when the half and its half-turn image form one symmetric meander shape.
	"""
	return isAxisMirrored吗(arcCode) and isJoined吗(makeLookupPartner(arcCode))

def doTheNeedful(state: StateMeandersAxis) -> StateMeandersAxis:
	"""Count the halves that join at the symmetry axis and add the count to `state.countAddend`.

	(AI generated docstring)

	You can use this function to complete a term of A223096. With D(w) the count of `arcCode` w at the
	symmetry axis, J the test `isHalfJoined吗`, and σ the reflection `reflectArcCode`, the term is Σ_w
	D(w)·(J(w) + J(σw)) / 2. J(σw) is the test with the center terminal on the side of `bitsZulu`.

	Parameters
	----------
	state : StateMeandersAxis
		Transfer state of kind `'pile'` before the transfer.

	Returns
	-------
	state : StateMeandersAxis
		The same state with the term in `state.countAddend` and an empty `state.lookupMeanders`. Read
		the term from `state.totalMeanders`.
	"""
	state = count(state)
	state.countAddend += sum(map(mul, state.lookupMeanders.values(), map(add
		, map(isHalfJoined吗, state.lookupMeanders.keys()), map(isHalfJoined吗, map(reflectArcCode, state.lookupMeanders.keys())), strict=True), strict=True)) // 2
	state.lookupMeanders = {}
	return state
