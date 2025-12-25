# ruff: noqa: T201
"""Analyze entropy of leaves in folding sequences to understand their distributional properties."""
from mapFolding._e.analysisPython.Z0Z_patternFinder import measureEntropy
from mapFolding._e.dataBaskets import EliminationState

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
	if leaf63 in dataframeEntropy['leaf'].values:
		row63 = dataframeEntropy[dataframeEntropy['leaf'] == leaf63].iloc[0]
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

		leavesWithSameBitCount = dataframeEntropy[dataframeEntropy['bitCount'] == row63['bitCount']]
		print(f"\nLeaves with same bit_count ({row63['bitCount']}):")
		print(leavesWithSameBitCount[['leaf', 'bitPattern', 'entropyRelative', 'concentrationMaximum']].head(10).to_string(index=False))

		leavesWithSameTrailingZeros = dataframeEntropy[dataframeEntropy['trailingZeros'] == row63['trailingZeros']]
		print(f"\nLeaves with same trailing zeros ({row63['trailingZeros']}):")
		print(leavesWithSameTrailingZeros[['leaf', 'bitPattern', 'entropyRelative', 'concentrationMaximum']].head(10).to_string(index=False))

		powersOf2 = dataframeEntropy[dataframeEntropy['bitCount'] == 1]
		print("\nPowers of 2 (bitCount == 1):")
		print(powersOf2[['leaf', 'bitPattern', 'entropyRelative', 'concentrationMaximum']].to_string(index=False))

	print(f"\n{'=' * 80}")
	print("Summary Statistics")
	print(f"{'=' * 80}")
	print(dataframeEntropy[['entropyRelative', 'concentrationMaximum']].describe())

if __name__ == '__main__':
	analyzeEntropyForDimension(6)
