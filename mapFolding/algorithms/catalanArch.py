"""Count noncrossing arch diagrams by the number of endpoint removals they survive.

(AI generated docstring)

You can obtain a histogram of reduction lengths or a row of cumulative counts. An arch diagram
uses a Dyck word with one bit per endpoint, read from the highest bit to the lowest bit.
Opening endpoints have bit value one, and closing endpoints have bit value zero.

Contents
--------
Functions
	advanceDyck
		Obtain the next Dyck word in increasing integer order.
	count
		Count diagrams for each exact reduction length.
	doTheNeedful
		Obtain a row of counts for successive reduction lengths.
	findPartnerFirst
		Locate the endpoint paired with the first endpoint.
	findPartnerLast
		Locate the endpoint paired with the last endpoint.
	generations
		Count diagram generations under successive endpoint removals.

Integer Types
	oneByte
		Numba integer type for reduction counters.
	two_n_bits
		Numba integer type for packed arch diagrams.
"""
from __future__ import annotations

from numba import int64, jit_module, uint8
from numba.core import types
from operator import neg

def count(n: int) -> list[int]:
	"""Count noncrossing diagrams of `n` arches for each exact reduction length.

	(AI generated docstring)

	Parameters
	----------
	n : int
		Positive number of arches. Packed diagrams must fit the configured integer type.

	Returns
	-------
	histogram : list[int]
		`n` counts. Position `k - 1` counts diagrams for which `generations` returns `k`.
	"""
	archCodeMaximum: int = ((1 << n) - 1) << n
	archCodeAlternating: int = ((1 << (2 * n)) - 1) // 3 * 2
	bitEndpointFirst: int = 1 << (2 * n - 1)
	archCode: int = archCodeAlternating
	histogram: list[int] = [0] * (n + 1)
	while archCode <= archCodeMaximum:  # C(n) iterations.
		histogram[generations(archCode, bitEndpointFirst)] += 1
		archCode = advanceDyck(archCode, archCodeAlternating)
	return histogram[1:None]

def generations(archCode: int, bitEndpointFirst: int) -> int:
	"""Count diagram generations until the first and last endpoints share an arch.

	(AI generated docstring)

	Each reduction removes the first and last endpoints and joins their former partners.
	The returned count includes the final diagram whose first and last endpoints are paired.

	Parameters
	----------
	archCode : int
		Nonempty Dyck word encoding a noncrossing arch diagram.
	bitEndpointFirst : int
		One-bit mask at the first endpoint, the highest bit of `archCode`.

	Returns
	-------
	generationsSurvived : int
		Number of diagrams in the reduction sequence, including the supplied diagram.
	"""
	generationsSurvived: int = 1
	bitPartnerFirst: int = findPartnerFirst(archCode, bitEndpointFirst)
	while 1 < bitPartnerFirst:
		archCode = (archCode ^ bitEndpointFirst ^ bitPartnerFirst ^ findPartnerLast(archCode)) >> 1
		bitEndpointFirst >>= 2
		generationsSurvived += 1
		bitPartnerFirst = findPartnerFirst(archCode, bitEndpointFirst)
	return generationsSurvived

def findPartnerFirst(archCode: int, bitEndpointFirst: int) -> int:
	"""Locate the endpoint paired with the first endpoint of an arch diagram.

	(AI generated docstring)

	Parameters
	----------
	archCode : int
		Nonempty Dyck word encoding a noncrossing arch diagram.
	bitEndpointFirst : int
		One-bit mask at the first endpoint, the highest bit of `archCode`.

	Returns
	-------
	bitPartnerFirst : int
		One-bit mask at the closing endpoint paired with the first endpoint.
	"""
	depth: int = 1
	while depth:
		bitEndpointFirst >>= 1
		depth += 2 * bool(archCode & bitEndpointFirst) - 1
	return bitEndpointFirst

def findPartnerLast(archCode: int) -> int:
	"""Locate the endpoint paired with the last endpoint of an arch diagram.

	(AI generated docstring)

	Parameters
	----------
	archCode : int
		Nonempty Dyck word encoding a noncrossing arch diagram.

	Returns
	-------
	bitPartnerLast : int
		One-bit mask at the opening endpoint paired with the lowest-bit endpoint.
	"""
	bitEndpointLast: int = 1
	depth: int = 1
	while depth:
		bitEndpointLast <<= 1
		depth += 1 - 2 * bool(archCode & bitEndpointLast)
	return bitEndpointLast

def advanceDyck(archCode: int, archCodeAlternating: int) -> int:
	"""Obtain the next Dyck word in increasing integer order.

	(AI generated docstring)

	Parameters
	----------
	archCode : int
		Nonempty Dyck word to advance.
	archCodeAlternating : int
		Alternating opening and closing bits for the same number of arches.

	Returns
	-------
	archCodeNext : int
		Next packed word. Advancing the last diagram exceeds the largest word of that length.
	"""
	bitPivot: int = archCode & neg(archCode)
	archCodePivoted: int = archCode + bitPivot
	pairsReset: int = (((archCode ^ archCodePivoted) // bitPivot) >> 2) + 1
	return ((pairsReset * pairsReset - 1) & archCodeAlternating) | archCodePivoted

def doTheNeedful(n: int) -> list[int]:
	"""Obtain row `n` of the arch-reduction triangle A287548.

	(AI generated docstring)

	Parameters
	----------
	n : int
		Positive number of arches. Packed diagrams must fit the configured integer type.

	Returns
	-------
	cells : list[int]
		`n` cumulative counts. Position `k - 1` counts diagrams with at least `k` generations.
		The first cell counts all noncrossing diagrams of `n` arches.

	See Also
	--------
	`count`
		Obtain counts for exact reduction lengths before taking cumulative sums.
	"""
	histogram: list[int] = count(n)
	return [sum(histogram[-(n - index):None]) for index in range(n)]

two_n_bits = int64
oneByte = uint8

jit_module(cache=True, error_model='numpy', fastmath=True, forceinline=True, locals={
	'archCode': two_n_bits,
	'archCodeAlternating': two_n_bits,
	'archCodeMaximum': two_n_bits,
	'archCodePivoted': two_n_bits,
	'bitEndpointFirst': two_n_bits,
	'bitEndpointLast': two_n_bits,
	'bitPartnerFirst': two_n_bits,
	'bitPivot': two_n_bits,
	'depth': oneByte,
	'generationsSurvived': oneByte,
	'histogram': types.List(int64),
	'index': oneByte,
	'n': oneByte,
	'pairsReset': two_n_bits,
})
