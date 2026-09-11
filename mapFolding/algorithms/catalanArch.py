from __future__ import annotations

from numba import int64, jit_module, types, uint8
from operator import neg

def count(n: int) -> list[int]:
	archCodeMaximum: int = ((1 << n) - 1) << n
	Z0Z_calibrator: int = ((1 << (2 * n)) - 1) // 3 * 2
	bitEndpointFirst: int = 1 << (2 * n - 1)
	archCode: int = Z0Z_calibrator
	histogram: list[int] = [0] * (n + 1)
	while archCode <= archCodeMaximum:  # C(n) iterations.
		histogram[generations(archCode, bitEndpointFirst)] += 1
		archCode = advanceDyck(archCode, Z0Z_calibrator)
	return histogram[1:]

def generations(archCode: int, bitEndpointFirst: int) -> int:
	generationsSurvived: int = 1
	bitPartnerFirst: int = findPartnerFirst(archCode, bitEndpointFirst)
	while 1 < bitPartnerFirst:
		archCode = (archCode ^ bitEndpointFirst ^ bitPartnerFirst ^ findPartnerLast(archCode)) >> 1
		bitEndpointFirst >>= 2
		generationsSurvived += 1
		bitPartnerFirst = findPartnerFirst(archCode, bitEndpointFirst)
	return generationsSurvived

def findPartnerFirst(archCode: int, bitEndpointFirst: int) -> int:
	depth: int = 1
	while depth:
		bitEndpointFirst >>= 1
		depth += 2 * bool(archCode & bitEndpointFirst) - 1
	return bitEndpointFirst

def findPartnerLast(archCode: int) -> int:
	bitEndpointLast: int = 1
	depth: int = 1
	while depth:
		bitEndpointLast <<= 1
		depth += 1 - 2 * bool(archCode & bitEndpointLast)
	return bitEndpointLast

def advanceDyck(archCode: int, archCodeAlternating: int) -> int:
	bitPivot: int = archCode & neg(archCode)
	archCodePivoted: int = archCode + bitPivot
	pairsReset: int = (((archCode ^ archCodePivoted) // bitPivot) >> 2) + 1
	return ((pairsReset * pairsReset - 1) & archCodeAlternating) | archCodePivoted

def doTheNeedful(n: int) -> list[int]:
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
	'Z0Z_calibrator': two_n_bits,
})
