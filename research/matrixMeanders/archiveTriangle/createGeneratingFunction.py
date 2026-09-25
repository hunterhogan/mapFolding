# DEVELOPMENT
# ruff: file-ignore[undocumented-public-module, undocumented-public-function, unnecessary-map, print]
# pyright: reportArgumentType=false
from __future__ import annotations

from functools import partial, reduce
from itertools import chain
from mapFolding.kitFilesystem import readDiagonal
from research.matrixMeanders.formulasTriangle._A005315 import _crunchDenominator, _crunchNumerator
from research.matrixMeanders.infoBooth import pathFilenameTriangleSemiCommaSeparatedValues
from sympy import mobius
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence

def cyclotomicMultiplicitiesToSteps(multiplicities: Mapping[int, int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
	if not multiplicities or min(multiplicities) < 1 or min(multiplicities.values()) < 0:
		message: str = f"I received `{multiplicities = }`, but I need positive cyclotomic orders and nonnegative multiplicities."
		raise ValueError(message)

	def calculateStepMultiplicity(step: int) -> tuple[int, int]:
		return step, sum(map(lambda multiple: multiplicities.get(step * multiple, 0) * int(mobius(multiple)),
			range(1, max(multiplicities) // step + 1)))

	stepsSigned: tuple[tuple[int, int], ...] = tuple(map(calculateStepMultiplicity, range(1, max(multiplicities) + 1)))
	return (
		tuple(chain.from_iterable(map(lambda step: (step[0],) * (-step[1]), filter(lambda step: step[1] < 0, stepsSigned))))
		, tuple(chain.from_iterable(map(lambda step: (step[0],) * step[1], filter(lambda step: 0 < step[1], stepsSigned))))
	)

def makeA005315Steps(次function: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
	if 次function < 2:
		message: str = f"I received `{次function = }`, but the denominator conjecture starts at A005315of2."
		raise ValueError(message)

	def calculateCyclotomicMultiplicity(order: int) -> tuple[int, int]:
		multiplicity: int
		if order == 1:
			multiplicity = 2 * 次function - 1
		elif order == 2:
			multiplicity = 2 * 次function - 3
		elif order == 3:
			multiplicity = 次function - 2
		else:
			multiplicity = max(0, 次function - (order // 2 if 8 <= order and order % 2 == 0 else order) + 1)
		return order, multiplicity

	return cyclotomicMultiplicitiesToSteps(dict(map(calculateCyclotomicMultiplicity, range(1, 2 * 次function + 1))))

def makeNumerator(sequence: Sequence[int], stepsNumerator: tuple[int, ...], stepsDenominator: tuple[int, ...], *,
	multiplier: int = 1, numeratorLength: int | None = None) -> tuple[int, ...]:
	if multiplier < 1 or any(map(lambda step: step < 1, stepsNumerator + stepsDenominator)):
		message: str = f"I received `{multiplier = }`, `{stepsNumerator = }`, and `{stepsDenominator = }`; I need positive integers."
		raise ValueError(message)
	if numeratorLength is None:
		numeratorLength = sum(stepsDenominator) - sum(stepsNumerator)
	if numeratorLength < 1 or len(sequence) < numeratorLength:
		message = f"I received `{len(sequence) = }` and `{numeratorLength = }`, but I need at least a positive numerator length of consecutive values."
		raise ValueError(message)
	numerator: tuple[int, ...] = reduce(_crunchNumerator, stepsDenominator, tuple(sequence[:numeratorLength]))
	numerator = reduce(partial(_crunchDenominator, 次coefficient=numeratorLength - 1), stepsNumerator, numerator)
	if any(map(lambda coefficient: coefficient % multiplier, numerator)):
		message = f"I received `{multiplier = }`, but it does not divide every reconstructed numerator coefficient."
		raise ValueError(message)
	return tuple(map(lambda coefficient: coefficient // multiplier, numerator))

def makeA005315Formula(次function: int, diagonal: Mapping[int, int], *, multiplier: int = 2
	) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
	stepsNumerator: tuple[int, ...]
	stepsDenominator: tuple[int, ...]
	stepsNumerator, stepsDenominator = makeA005315Steps(次function)
	return (
		makeNumerator(tuple(map(diagonal.__getitem__, range(2 * 次function,
			2 * 次function + sum(stepsDenominator) - sum(stepsNumerator)))), stepsNumerator, stepsDenominator, multiplier=multiplier)
		, stepsNumerator
		, stepsDenominator
	)

if __name__ == '__main__':
	次function: int = 3

	numerator, stepsNumerator, stepsDenominator = makeA005315Formula(次function,
		readDiagonal(pathFilenameTriangleSemiCommaSeparatedValues, 次function)
		# | readDiagonal(makePathFilenameDiagonal(次function), 次function, formatData='diagonalCSV')
	)
	message: str = f"{numerator = }\n{stepsNumerator = }\n{stepsDenominator = }"
	print(message)
