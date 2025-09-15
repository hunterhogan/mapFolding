"""Buckets."""
from mapFolding import MatrixMeandersState
from mapFolding.reference.A005316facts import bucketsIf_k_EVEN_by_nLess_k, bucketsIf_k_ODD_by_nLess_k
from math import exp, log
from typing import NamedTuple
import math

class ImaKey(NamedTuple):
	"""keys for dictionaries."""

	oeisID: str
	kIsOdd: bool
	nLess_kIsOdd: bool

def getBucketsTotal(state: MatrixMeandersState, safetyMultiplicand: float = 1.2) -> int:
	"""Estimate the total number of non-unique curveLocations that will be computed from the existing curveLocations.

	Notes
	-----
	Subexponential bucketsTotal unified estimator parameters (derived in reference notebook).

	The model is: log(buckets) = intercept + bN*log(n) + bK*log(k) + bD*log(n-k) + g_r*(k/n) + g_r2*(k/n)^2 + g_s*((n-k)/n) + offset(subseries)
	Subseries key: f"{oeisID}_kOdd={int(kIsOdd)}_dOdd={int(nLess_kIsOdd)}" with a reference subseries offset of zero.
	These coefficients intentionally remain in-source (SSOT) to avoid runtime JSON parsing overhead and to support reproducibility.
	"""
	dictionaryExponentialCoefficients: dict[ImaKey, float] = {
		(ImaKey(oeisID='', kIsOdd=False, nLess_kIsOdd=True)): 0.834,
		(ImaKey(oeisID='', kIsOdd=False, nLess_kIsOdd=False)): 1.5803,
		(ImaKey(oeisID='', kIsOdd=True, nLess_kIsOdd=True)): 1.556,
		(ImaKey(oeisID='', kIsOdd=True, nLess_kIsOdd=False)): 1.8047,
	}

	logarithmicOffsets: dict[ImaKey, float] ={
		(ImaKey('A000682', kIsOdd=False, nLess_kIsOdd=False)): 0.0,
		(ImaKey('A000682', kIsOdd=False, nLess_kIsOdd=True)): -0.07302547148212568,
		(ImaKey('A000682', kIsOdd=True, nLess_kIsOdd=False)): -0.00595307513938792,
		(ImaKey('A000682', kIsOdd=True, nLess_kIsOdd=True)): -0.012201222865243722,
		(ImaKey('A005316', kIsOdd=False, nLess_kIsOdd=False)): -0.6392728422078733,
		(ImaKey('A005316', kIsOdd=False, nLess_kIsOdd=True)): -0.6904925299923548,
		(ImaKey('A005316', kIsOdd=True, nLess_kIsOdd=False)): 0.0,
		(ImaKey('A005316', kIsOdd=True, nLess_kIsOdd=True)): 0.0,
	}

	logarithmicParameters: dict[str, float] = {
		'intercept': -166.1750299793178,
		'log(n)': 1259.0051001675547,
		'log(k)': -396.4306071056408,
		'log(nLess_k)': -854.3309503739766,
		'k/n': 716.530410654819,
		'(k/n)^2': -2527.035113444166,
		'normalized k': -882.7054406339189,
	}

	bucketsTotalMaximumBy_kOfMatrix: dict[int, int] = {1:3, 2:12, 3:40, 4:125, 5:392, 6:1254, 7:4087, 8:13623, 9:46181, 10:159137, 11:555469, 12:1961369, 13:6991893, 14:25134208}

	xCommon = 1.57

	nLess_k: int = state.n - state.kOfMatrix
	kIsOdd: bool = bool(state.kOfMatrix & 1)
	nLess_kIsOdd: bool = bool(nLess_k & 1)
	kIsEven: bool = not kIsOdd
	bucketsTotal: int = -8

	"""NOTE temporary notes
	I have a fault in my thinking. bucketsTotal increases as k decreases until ~0.4k, then bucketsTotal decreases rapidly to 1. I
	have ignored the decreasing side. In the formulas for estimation, I didn't differentiate between increasing and decreasing.
	So, I probably need to refine the formulas. I guess I need to add checks to the if/else monster, too.

	While buckets is increasing:
		3 types of estimates:
			1. Exponential growth.
			2. Logarithmic growth.
			3. Hard ceiling.
	While buckets is decreasing:
		1. Hard ceiling, same as increasing side.
		2. ???
		3. buckets = 1.

	The formula for exponential growth _never_ underestimates. I haven't measured by how much it overestimates.

	"""

	bucketsTotalAtMaximum: bool = state.kOfMatrix <= ((state.n - 1 - (state.kOfMatrix % 2)) // 3)
	bucketsTotalGrowsExponentially: bool = state.kOfMatrix > nLess_k
	bucketsTotalGrowsLogarithmically: bool = state.kOfMatrix > ((state.n - (state.n % 3)) // 3)

	if bucketsTotalAtMaximum:
		if state.kOfMatrix in bucketsTotalMaximumBy_kOfMatrix:
			bucketsTotal = bucketsTotalMaximumBy_kOfMatrix[state.kOfMatrix]
		else:
			c = 0.95037
			r = 3.3591258254
			if kIsOdd:
				c = 0.92444
				r = 3.35776
			bucketsTotal = int(c * r**state.kOfMatrix * safetyMultiplicand)

	elif bucketsTotalGrowsExponentially:
		if (state.oeisID == 'A005316') and kIsOdd and (nLess_k in bucketsIf_k_ODD_by_nLess_k):
			# If I already know bucketsTotal.
			bucketsTotal = bucketsIf_k_ODD_by_nLess_k[nLess_k]
		elif (state.oeisID == 'A005316') and kIsEven and (nLess_k in bucketsIf_k_EVEN_by_nLess_k):
			# If I already know bucketsTotal.
			bucketsTotal = bucketsIf_k_EVEN_by_nLess_k[nLess_k]
		else: # I estimate bucketsTotal during exponential growth.
			xInstant: int = math.ceil(nLess_k / 2)
			A000682adjustStartingCurveLocations: float = 0.25
			startingConditionsCoefficient: float = dictionaryExponentialCoefficients[ImaKey('', kIsOdd, nLess_kIsOdd)]
			if kIsOdd and nLess_kIsOdd:
				A000682adjustStartingCurveLocations = 0.0
			if state.oeisID == 'A000682': # NOTE Net effect is between `*= n` and `*= n * 2.2` if n=46.
				startingConditionsCoefficient *= state.n * (((state.n // 2) + 2) ** A000682adjustStartingCurveLocations)
			bucketsTotal = int(startingConditionsCoefficient * math.exp(xCommon * xInstant))

	elif state.kOfMatrix <= max(bucketsTotalMaximumBy_kOfMatrix.keys()):
		# If `kOfMatrix` is low, use maximum bucketsTotal. 1. Can't underestimate. 2. Skip computation that can underestimate.
		# 3. The potential difference in memory use is relatively small.
		bucketsTotal = bucketsTotalMaximumBy_kOfMatrix[state.kOfMatrix]

	elif bucketsTotalGrowsLogarithmically:
		xPower: float = (0
			+ logarithmicParameters['intercept']
			+ logarithmicParameters['log(n)'] * log(state.n)
			+ logarithmicParameters['log(k)'] * log(state.kOfMatrix)
			+ logarithmicParameters['log(nLess_k)'] * log(nLess_k)
			+ logarithmicParameters['k/n'] * (state.kOfMatrix / state.n)
			+ logarithmicParameters['(k/n)^2'] * (state.kOfMatrix / state.n)**2
			+ logarithmicParameters['normalized k'] * nLess_k / state.n
			+ logarithmicOffsets[ImaKey(state.oeisID, kIsOdd, nLess_kIsOdd)]
		)

		bucketsTotal = int(exp(xPower * safetyMultiplicand))

	else:
		message = "I shouldn't be here."
		raise SystemError(message)
	return bucketsTotal
