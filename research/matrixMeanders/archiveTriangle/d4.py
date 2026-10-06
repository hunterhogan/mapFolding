# ruff: file-ignore[implicit-namespace-package, undocumented-public-module, undocumented-public-class, undocumented-public-function]

from __future__ import annotations

from fractions import Fraction
from functools import partial, reduce
from typing import NamedTuple

class PeriodicCoefficients(NamedTuple):
	period: int
	byResidue: tuple[Fraction, ...]

boxOfDegreesDescending: tuple[PeriodicCoefficients, PeriodicCoefficients, PeriodicCoefficients
							, PeriodicCoefficients, PeriodicCoefficients, PeriodicCoefficients
							, PeriodicCoefficients] = (
	PeriodicCoefficients(1, (Fraction(49, 5 * 12**2 * 4 * 8),))
	, PeriodicCoefficients(1, (Fraction(1171, 5 * 12**2 * 4**2),))
	, PeriodicCoefficients(2, (Fraction(6893, 9 * 12 * 4 * 8)
						, Fraction(27617, 9 * 12 * 4**2 * 8)))
	, PeriodicCoefficients(2, (Fraction(3625, 12 * 4**2)
						, Fraction(7279, 12 * 4 * 8)))
	, PeriodicCoefficients(2, (Fraction(11929, 5 * 4 * 8)
						, Fraction(578357, 5 * 12 * 4**2 * 8)))
	, PeriodicCoefficients(6, (Fraction(6697, 5 * 12)
						, Fraction(4004017, 5 * 9 * 12 * 8**2)
						, Fraction(60353, 5 * 9 * 12)
						, Fraction(443753, 5 * 12 * 8**2)
						, Fraction(60433, 5 * 9 * 12)
						, Fraction(3998897, 5 * 9 * 12 * 8**2)))
	, PeriodicCoefficients(24, (
		Fraction(42, 1)
		, Fraction(229381, 9 * (2**3)**3)
		, Fraction(3161, 9 * 2**3)
		, Fraction(24413, (2**3)**3)
		, Fraction(787, 9 * 2)
		, Fraction(229637, 9 * (2**3)**3)
		, Fraction(333, 2**3)
		, Fraction(229957, 9 * (2**3)**3)
		, Fraction(394, 9)
		, Fraction(24349, (2**3)**3)
		, Fraction(3193, 9 * 2**3)
		, Fraction(227909, 9 * (2**3)**3)
		, Fraction(83, 2)
		, Fraction(231685, 9 * (2**3)**3)
		, Fraction(3125, 9 * 2**3)
		, Fraction(24413, (2**3)**3)
		, Fraction(398, 9)
		, Fraction(227333, 9 * (2**3)**3)
		, Fraction(337, 2**3)
		, Fraction(229957, 9 * (2**3)**3)
		, Fraction(779, 9 * 2)
		, Fraction(24605, (2**3)**3)
		, Fraction(3157, 9 * 2**3)
		, Fraction(227909, 9 * (2**3)**3)
	))
)

def _accumulate(accumulation: Fraction, accretion: PeriodicCoefficients, index: int) -> Fraction:
	return accumulation * index + accretion.byResidue[index % accretion.period]

def diagonal4(x: int) -> int:
	return int(reduce(partial(_accumulate, index=x - 2 * 4), boxOfDegreesDescending, Fraction(0)))
