from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding.theTypes import Folding, Leaf, Pile

def foldingValid吗(folding: Folding, mapShape: tuple[int, ...]) -> bool:
	lookupPile: dict[Leaf, Pile] = dict(zip(folding, range(len(folding)), strict=True))
	dimension: int = 0
	dimensionStride: int = 1
	foldingValid: bool = True
	while dimension < len(mapShape) and foldingValid:
		stacksByParity: tuple[list[Leaf], list[Leaf]] = ([], [])
		pile: Pile = 0
		while pile < len(folding) and foldingValid:
			leaf: Leaf = folding[pile]
			leafCoordinate: int = (leaf // dimensionStride) % mapShape[dimension]
			if leafCoordinate <= mapShape[dimension]:
				leafCreasePost: Leaf = leaf + dimensionStride
				if pile < lookupPile[leafCreasePost]:
					stacksByParity[leafCoordinate & 1].append(leaf)
				else:
					foldingValid = stacksByParity[leafCoordinate & 1].pop() == leafCreasePost
			if 0 < leafCoordinate and foldingValid:
				leafCreaseAnte: Leaf = leaf - dimensionStride
				if pile < lookupPile[leafCreaseAnte]:
					stacksByParity[(leafCoordinate - 1) & 1].append(leaf)
				else:
					foldingValid = stacksByParity[(leafCoordinate - 1) & 1].pop() == leafCreaseAnte
			pile += 1
		dimensionStride *= mapShape[dimension]
		dimension += 1
	return foldingValid
