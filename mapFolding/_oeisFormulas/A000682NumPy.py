"""Semi-meanders."""
from mapFolding._oeisFormulas.matrixMeanders import count
from mapFolding._oeisFormulas.matrixMeanders64 import count64
from mapFolding._oeisFormulas.matrixMeandersMimic import countMimic
import gc

# TODO merge `matrixMeanders.count` and `matrixMeandersMimic.countMimic`.
# BUT, the thing that has prevented me from merging them before is `A005316` also calls `matrixMeanders.count`.
# And when I make the changes that I think should work, `A005315`, not `A005316`, breaks.

def initializeA000682(n: int) -> dict[int, int]:
	curveLocationsMAXIMUM: int = 1 << (2 * n + 4)

	curveSeed: int = 5 - (n & 0b1) * 4
	listCurveLocations: list[int] = [(curveSeed << 1) | curveSeed]

	while listCurveLocations[-1] < curveLocationsMAXIMUM:
		curveSeed = (curveSeed << 4) | 0b101
		listCurveLocations.append((curveSeed << 1) | curveSeed)

	return dict.fromkeys(listCurveLocations, 1)

def A000682(n: int) -> int:
	"""Compute a(n) for OEIS ID A000682.

	Parameters
	----------
	n : int
		The index in the OEIS ID sequence.

	Returns
	-------
	a(n) : int
		The computed value of a(n).

	Making sausage
	--------------

	As first computed by Iwan Jensen in 2000, a(41) = 6664356253639465480.
	Citation: https://github.com/hunterhogan/mapFolding/blob/main/citations/Jensen.bibtex
	See also https://oeis.org/A000682

	I'm sure you instantly observed that a(41) = 6664356253639465480.bit_length() = 63 bits.

	If you ask NumPy 2.3, "What is your relationship with integers with more than 64 bits?"
	NumPy will say, "It's complicated."

	Therefore, to take advantage of the computational excellence of NumPy when computing a(n) for n > 41, I must make some
	adjustments at the total count approaches 64 bits.

	The second complication is bit-packed integers. I use a loop that starts at `bridges = n` and decrements (`bridges -= 1`)
	`until bridges = 0`. If `bridges > 29`, some of the bit-packed integers have more than 64 bits. "Hey NumPy, can I use
	bit-packed integers with more than 64 bits?" NumPy: "It's complicated." Therefore, while `bridges` is decrementing, I don't
	use NumPy until I believe the bit-packed integers will be less than 64 bits.

	A third fact that works in my favor is that peak memory usage occurs when all of the integers are well under 64 bits wide.

	In total, to compute a(n) for "large" n, I use three-stages.
	1. I use Python primitive `int` contained in a Python primitive `dict`.
	2. When the bit width of `bridges` is small enough to use `numpy.uint64`, switch to `numpy` for the heavy lifting.
	3. When distinctCrossings subtotals exceed 64 bits, I must switch back to Python primitives.
	"""
# NOTE '29' is based on two things. 1) `bridges = 29`, groupZuluLocator = 0xaaaaaaaaaaaaaaaa.bit_length() = 64. 2) If `bridges =
# 30` or a larger number, `OverflowError: int too big to convert`. Conclusion: '29' isn't necessarily correct or the best value:
# it merely fits within my limited ability to assess the correct value.
	count64_bridgesMaximum = 29
	bridgesMinimum = 0  # NOTE This default value is necessary: it prevents `count64` from returning an incomplete dictionary when that is not necessary.

	distinctCrossings64bitLimitAsValueOf_n = 41
	distinctCrossingsSubtotal64bitLimitAsValueOf_n_WAG = distinctCrossings64bitLimitAsValueOf_n - 2
	distinctCrossings64bitLimitSafetyMargin = 3
	# Oh, uh, I suddenly had an intuition that this method of computing 64bitLimitAsValueOf_n is, at best, wrong.
	dictionaryCurveLocations: dict[int, int] = initializeA000682(n - 1)
	if n >= count64_bridgesMaximum:
		if n >= distinctCrossingsSubtotal64bitLimitAsValueOf_n_WAG:
			bridgesMinimum = n - distinctCrossingsSubtotal64bitLimitAsValueOf_n_WAG + distinctCrossings64bitLimitSafetyMargin
		n, dictionaryCurveLocations = countMimic(n - 1, dictionaryCurveLocations, count64_bridgesMaximum)
		n += 1
		gc.collect()
	n, dictionaryCurveLocations = count64(n - 1, dictionaryCurveLocations, bridgesMinimum)
	if n > 0:
		print('Stage 3: `count`')
		n += 1
		gc.collect()
		return count(n - 1, dictionaryCurveLocations)
	return sum(dictionaryCurveLocations.values())

