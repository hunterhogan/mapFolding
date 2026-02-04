# ruff: noqa: T201
"""Analyze entropy of leaves in folding sequences to understand their distributional properties."""
from hunterMakesPy import raiseIfNone
from mapFolding._e import dimensionNearestTail, getLeafDomain, pileOrigin, 零
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.dataDynamic import getDataFrameFoldings
from typing import Any, TYPE_CHECKING
import numpy
import pandas

if TYPE_CHECKING:
	from pandas import DataFrame, Series

def measureEntropy(state: EliminationState, listLeavesAnalyzed: list[int] | None = None) -> pandas.DataFrame:
	"""Measure the relative entropy and distributional properties of leaves across folding sequences.

	This function analyzes how leaves are distributed across their mathematical domains by comparing
	empirical distributions from actual folding sequences against uniform distributions. The analysis
	uses Shannon entropy normalized by maximum possible entropy to produce comparable measures across
	leaves with different domain sizes.

	Parameters
	----------
	state : EliminationState
		The elimination state containing the map shape and dimension information.
	listLeavesAnalyzed : list[int] | None = None
		Specific leaves to analyze. If None, analyzes all leaves except the trivial ones
		(0, 1, and leavesTotal-1) which always occupy the same pile.

	Returns
	-------
	dataframeEntropy : pandas.DataFrame
		DataFrame with columns:
		- 'leaf': The leaf value being analyzed
		- 'domainSize': Number of possible piles where this leaf can appear
		- 'entropyActual': Shannon entropy of the empirical distribution
		- 'entropyMaximum': Maximum possible entropy (uniform distribution)
		- 'entropyRelative': entropyActual / entropyMaximum (0 to 1)
		- 'concentrationMaximum': Maximum frequency / mean frequency
		- 'bitPattern': Binary representation for easy identification of patterns
		- 'bitCount': Number of 1s in binary representation
		- 'trailingZeros': Number of trailing zeros (power of 2 factor)
		Sorted by entropyRelative descending to show most uniform distributions first.

	Notes
	-----
	The relative entropy metric allows fair comparison between leaves with vastly different domain
	sizes. A value near 1.0 indicates nearly uniform distribution (high entropy, unpredictable),
	while values near 0.0 indicate highly concentrated distribution (low entropy, predictable).

	The concentration metric shows how peaked the distribution is by comparing the most frequent
	position to the mean frequency. Higher values indicate more predictable placement.

	"""
	dataframeFoldings: pandas.DataFrame = raiseIfNone(getDataFrameFoldings(state))

	if listLeavesAnalyzed is None:
		leavesExcluded: set[int] = {pileOrigin, 零, state.leavesTotal - 零}
		listLeavesAnalyzed = [leaf for leaf in range(state.leavesTotal) if leaf not in leavesExcluded]

	listEntropyRecords: list[dict[str, Any]] = []

	for leaf in listLeavesAnalyzed:
		domainLeaf: range = getLeafDomain(state, leaf)
		domainSize: int = len(domainLeaf)

		if domainSize == 0:
			continue

		dataframeMelted: pandas.DataFrame = dataframeFoldings[dataframeFoldings == leaf].melt(ignore_index=False)
		dataframeMelted = dataframeMelted.dropna()
		if dataframeMelted.empty:
			continue

		arrayPileCounts: numpy.ndarray = numpy.bincount(dataframeMelted['variable'].astype(int), minlength=state.leavesTotal)
		arrayPileCountsInDomain: numpy.ndarray = arrayPileCounts[list(domainLeaf)]
		arrayFrequencies: numpy.ndarray = arrayPileCountsInDomain / arrayPileCountsInDomain.sum()

		maskNonzero: numpy.ndarray = arrayFrequencies > 0
		entropyActual: float = float(-numpy.sum(arrayFrequencies[maskNonzero] * numpy.log2(arrayFrequencies[maskNonzero])))
		entropyMaximum: float = float(numpy.log2(domainSize))
		entropyRelative: float = entropyActual / entropyMaximum if entropyMaximum > 0 else 0.0

		frequencyMaximum: float = float(arrayFrequencies.max())
		frequencyMean: float = 1.0 / domainSize
		concentrationMaximum: float = frequencyMaximum / frequencyMean if frequencyMean > 0 else 0.0

		listEntropyRecords.append({
			'leaf': leaf,
			'domainSize': domainSize,
			'entropyActual': entropyActual,
			'entropyMaximum': entropyMaximum,
			'entropyRelative': entropyRelative,
			'concentrationMaximum': concentrationMaximum,
			'bitPattern': leaf.__format__('06b'),
			'bitCount': leaf.bit_count(),
			'trailingZeros': dimensionNearestTail(leaf),
		})

	return pandas.DataFrame(listEntropyRecords).sort_values('entropyRelative', ascending=False).reset_index(drop=True)

def analyzeEntropyForDimension(dimensionsTotal: int = 6) -> None:
	"""Analyze entropy for all non-trivial leaves in a given dimension configuration."""
	mapShape: tuple[int, ...] = (2,) * dimensionsTotal
	state: EliminationState = EliminationState(mapShape)

	print(f"\n{'=' * 80}")
	print(f"Entropy Analysis for 2^{dimensionsTotal} (mapShape={mapShape})")
	print(f"{'=' * 80}\n")

	dataframeEntropy = measureEntropy(state)

	print("\nTop 20 leaves by relative entropy (most uniform distributions):")
	print(dataframeEntropy.head(20).to_string(index=False))

	print("\n\nBottom 20 leaves by relative entropy (most concentrated distributions):")
	print(dataframeEntropy.tail(20).to_string(index=False))

	leaf63: int = 63
	if leaf63 in dataframeEntropy['leaf'].to_numpy():
		row63: Series = dataframeEntropy[dataframeEntropy['leaf'] == leaf63].iloc[0]
		rankOf63: int = int(dataframeEntropy[dataframeEntropy['leaf'] == leaf63].index[0])
		totalLeaves: int = len(dataframeEntropy)

		print(f"\n\n{'=' * 80}")
		print(f"Analysis of Leaf 63 ({bin(leaf63)} = 2^6 - 1, all dimensions have value 1)")
		print(f"{'=' * 80}")
		print(f"Rank: {rankOf63 + 1} / {totalLeaves}")
		print(f"Domain Size: {row63['domainSize']}")
		print(f"Entropy (actual): {row63['entropyActual']:.4f}")
		print(f"Entropy (maximum): {row63['entropyMaximum']:.4f}")
		print(f"Relative Entropy: {row63['entropyRelative']:.4f} (1.0 = uniform, 0.0 = concentrated)")
		print(f"Concentration: {row63['concentrationMaximum']:.4f}x mean frequency")
		print(f"Bit Count: {row63['bitCount']}")
		print(f"Trailing Zeros: {row63['trailingZeros']}")

		print("\n\nComparing leaf 63 to leaves with similar properties:")

		bitCount: int = int(row63['bitCount'])
		leavesWithSameBitCount: DataFrame = dataframeEntropy[dataframeEntropy['bitCount'] == bitCount]
		print(f"\nLeaves with same bit_count ({bitCount}):")
		subsetBitCount: DataFrame = leavesWithSameBitCount[['leaf', 'bitPattern', 'entropyRelative', 'concentrationMaximum']]
		print(subsetBitCount.head(10).to_string(index=False))

		trailingZeros: int = int(row63['trailingZeros'])
		leavesWithSameTrailingZeros: DataFrame = dataframeEntropy[dataframeEntropy['trailingZeros'] == trailingZeros]
		print(f"\nLeaves with same trailing zeros ({trailingZeros}):")
		subsetTrailing: DataFrame = leavesWithSameTrailingZeros[['leaf', 'bitPattern', 'entropyRelative', 'concentrationMaximum']]
		print(subsetTrailing.head(10).to_string(index=False))

		powersOf2: DataFrame = dataframeEntropy[dataframeEntropy['bitCount'] == 1]
		print("\nPowers of 2 (bitCount == 1):")
		subsetPowers: DataFrame = powersOf2[['leaf', 'bitPattern', 'entropyRelative', 'concentrationMaximum']]
		print(subsetPowers.to_string(index=False))

	print(f"\n{'=' * 80}")
	print("Summary Statistics")
	print(f"{'=' * 80}")
	print(dataframeEntropy[['entropyRelative', 'concentrationMaximum']].describe())

if __name__ == '__main__':
	analyzeEntropyForDimension(6)
