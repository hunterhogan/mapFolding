from __future__ import annotations

from mapFolding.algorithms.matrixAxisStrip import isJoined吗
from mapFolding.algorithms.matrixMeanders import count
from operator import add, mul
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterator
	from mapFolding.dataBaskets import StateMeandersAxis

def getBitsLocator(arcCode: int) -> int:
	return (1 << (arcCode.bit_length() | 1) + 1) // 3

def reflectArcCode(arcCode: int) -> int:
	bitsLocator: int = getBitsLocator(arcCode)
	return (arcCode & bitsLocator) << 1 | arcCode >> 1 & bitsLocator

def pairTerminals(bitsSide: int, boxOfTerminals: Iterator[int], lookupPartner: list[int]) -> list[int]:
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

def makeLookupPartner(arcCode: int) -> list[int]:
	bitsLocator: int = getBitsLocator(arcCode)
	bitsAlfa: int = arcCode & bitsLocator
	bitsZulu: int = arcCode >> 1 & bitsLocator
	lookupPartner: list[int] = list(range((bitsAlfa.bit_length() + bitsZulu.bit_length()) // 2 - 1))
	boxOfCrossAlfa: list[int] = pairTerminals(bitsAlfa, iter(range(len(lookupPartner))), lookupPartner)
	boxOfCrossZulu: list[int] = pairTerminals(bitsZulu, iter(range(len(lookupPartner) - 1, -1, -1)), lookupPartner)
	tuple(map(lookupPartner.__setitem__, boxOfCrossAlfa, boxOfCrossZulu, strict=True))
	tuple(map(lookupPartner.__setitem__, boxOfCrossZulu, boxOfCrossAlfa, strict=True))
	return lookupPartner

def isAxisMirrored吗(arcCode: int) -> bool:
	bitsLocator: int = getBitsLocator(arcCode)
	return (arcCode >> 1 & bitsLocator).bit_length() <= (arcCode & bitsLocator).bit_length() <= (arcCode >> 1 & bitsLocator).bit_length() + 4

def isHalfJoined吗(arcCode: int) -> bool:
	return isAxisMirrored吗(arcCode) and isJoined吗(makeLookupPartner(arcCode))

def doTheNeedful(state: StateMeandersAxis) -> StateMeandersAxis:
	state = count(state)
	state.countAddend += sum(map(mul, state.lookupMeanders.values(), map(add
		, map(isHalfJoined吗, state.lookupMeanders.keys()), map(isHalfJoined吗, map(reflectArcCode, state.lookupMeanders.keys())), strict=True), strict=True)) // 2
	state.lookupMeanders = {}
	return state
