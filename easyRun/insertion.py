from __future__ import annotations

from mapFolding.algorithms.eliminationCrease import doTheNeedful as eliminateFolds
from mapFolding.algorithms.insertion import doTheNeedful
from mapFolding.algorithms.insertion2上nDimensional import makeAlbum2上nDimensional, recordAlbum2上nDimensional
from mapFolding.beDRY import defineProcessorLimit
from mapFolding.dataBaskets import StateElimination
from mapFolding.reduceIt import boxOfFunctionsReduction2上nDimensional
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from hunterMakesPy import ConcurrencyLimit
	from pathlib import Path

if __name__ == '__main__':
	mapShape: tuple[int, ...] = (2, 3)
	CPUlimit: ConcurrencyLimit = -2
	start: float = perf_counter()

	if mapShape == (2,) * len(mapShape):
		workersMaximum: int = defineProcessorLimit(CPUlimit)
		state: StateElimination = StateElimination(mapShape, boxOfFunctionsReduction=boxOfFunctionsReduction2上nDimensional)
		state = makeAlbum2上nDimensional(state, workersMaximum)
		state.moveToBoxOfFolding()
		if state.boxOfPermutationSpace:
			state = eliminateFolds(state, workersMaximum)
		pathFilenameAlbum: Path = recordAlbum2上nDimensional(state)
	else:
		pathFilenameAlbum = doTheNeedful(mapShape, CPUlimit)

	print(f"{perf_counter() - start:.2f}")
	print(pathFilenameAlbum)
