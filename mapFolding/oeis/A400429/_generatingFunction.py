from __future__ import annotations

from functools import reduce
from itertools import chain, tee
from operator import mul
from sympy import cyclotomic_poly, divisors, mobius, Poly, Symbol

#=SIN= Ruff unnecessary-map suppression: the Python generation instructions prohibit generator comprehensions.
# ruff: file-ignore[unnecessary-map]
#=SIN= Pyright unknown-type suppressions: SymPy provides no type stubs for its exact polynomial operations.
# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false

def reduceFraction(numerator: tuple[int, ...], stepsNumerator: tuple[int, ...], stepsDenominator: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
	if any(map(lambda step: step < 1, chain(stepsNumerator, stepsDenominator))):
		message: str = f"I received `{stepsNumerator = }` and `{stepsDenominator = }`, but I need positive integer steps."
		raise ValueError(message)

	variable: Symbol = Symbol('z')

	def makePolynomial(step: int) -> Poly:  # pyright: ignore[reportUnknownParameterType]
		return Poly(1 - variable**step, variable, domain='ZZ')  # pyright: ignore[reportOperatorIssue]

	polynomialNumerator: Poly = reduce(mul, map(makePolynomial, stepsNumerator), Poly.from_list(numerator[::-1], variable, domain='ZZ'))
	polynomialDenominator: Poly = reduce(mul, map(makePolynomial, stepsDenominator), Poly(1, variable, domain='ZZ'))
	polynomialCommonFactor: Poly = polynomialNumerator.gcd(polynomialDenominator)
	polynomialNumerator = polynomialNumerator.exquo(polynomialCommonFactor)
	polynomialDenominator = polynomialDenominator.exquo(polynomialCommonFactor)
	polynomialNumerator = polynomialNumerator.mul_ground(polynomialDenominator.TC())
	polynomialDenominator = polynomialDenominator.mul_ground(polynomialDenominator.TC())
	return (tuple(map(int, reversed(polynomialNumerator.all_coeffs()))), tuple(map(int, reversed(polynomialDenominator.all_coeffs()))))

def reduceSteps(numerator: tuple[int, ...], stepsNumerator: tuple[int, ...], stepsDenominator: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
	numerator, denominator = reduceFraction(numerator, stepsNumerator, stepsDenominator)
	variable: Symbol = Symbol('z')

	def orderFactors(factorMultiplicity: tuple[Poly, int]) -> tuple[int, int]:  # pyright: ignore[reportUnknownParameterType]
		return (
			next(filter(lambda order: factorMultiplicity[0] == Poly(cyclotomic_poly(order, variable), variable, domain='ZZ'),
				chain.from_iterable(map(divisors, stepsDenominator))))
			, factorMultiplicity[1]
		)

	#=SIN= Intermediate mapping: Möbius inversion requires repeated lookup of the reduced cyclotomic multiplicities.
	multiplicities: dict[int, int] = dict(map(orderFactors, Poly.from_list(denominator[::-1], variable, domain='ZZ').factor_list()[1]))

	def getSign(step: int) -> tuple[int, int]:
		return step, sum(map(lambda order: multiplicities[order] * int(mobius(order // step)),  # pyright: ignore[reportArgumentType]
			filter(lambda order: order % step == 0, multiplicities)))

	stepsForNumerator, stepsForDenominator = tee(map(getSign, range(1, max(multiplicities, default=0) + 1)))
	return (
		numerator
		, tuple(chain.from_iterable(map(lambda stepMultiplicity: (stepMultiplicity[0],) * (-stepMultiplicity[1]),
			filter(lambda stepMultiplicity: stepMultiplicity[1] < 0, stepsForNumerator))))
		, tuple(chain.from_iterable(map(lambda stepMultiplicity: (stepMultiplicity[0],) * stepMultiplicity[1],
			filter(lambda stepMultiplicity: 0 < stepMultiplicity[1], stepsForDenominator))))
	)
