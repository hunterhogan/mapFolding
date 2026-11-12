"""Assert contracts about mapFolding's own domain objects, which `hunterMakesPy.tests` cannot depend on.

The shared assertion helpers come from `hunterMakesPy.tests` through `mapFolding.tests`.
"""
from __future__ import annotations

from hunterMakesPy.tests import messageTestFailure
from operator import methodcaller
from typing import TYPE_CHECKING
import numpy
import numpy.testing
import pandas

if TYPE_CHECKING:
	from collections.abc import Sequence
	from mapFolding.dataBaskets import PermutationSpace
	from numpy.typing import NDArray
	from typing import Any

def assertPermutationSpacesPartitionFoldings(boxOfPermutationSpace: Sequence[PermutationSpace], arrayFoldings: NDArray[numpy.uint8]
	, function: str, *arguments: Any, **keywordArguments: Any) -> None:
	"""Assert that every folding agrees with the pinned leaves of exactly one permutation space.

	Each row of `arrayFoldings` is one folding, and column `pile` holds the leaf at `pile`. A folding agrees with a permutation
	space when the folding has every leaf that the permutation space pins, at the pile where the permutation space pins it. A
	permutation space that pins no leaf agrees with every folding.
	"""
	leavesPinned: pandas.DataFrame = (pandas.DataFrame(map(methodcaller('pinnedLeaves'), boxOfPermutationSpace))
		.melt(ignore_index=False, var_name='pile', value_name='leaf').dropna().astype({'leaf': int}).reset_index(names='permutationSpace'))
	foldings: pandas.DataFrame = (pandas.DataFrame(arrayFoldings)
		.melt(ignore_index=False, var_name='pile', value_name='leaf').astype({'leaf': int}).reset_index(names='folding'))
	agreements: pandas.Series = leavesPinned.merge(foldings, on=['pile', 'leaf']).value_counts(['permutationSpace', 'folding'])
	agreementsRequired: pandas.Series = leavesPinned.value_counts('permutationSpace')
	agreementsComplete: pandas.Series = agreements[agreements.to_numpy() == agreementsRequired.reindex(agreements.index.get_level_values('permutationSpace')).to_numpy()]
	permutationSpacesPinningNothing: int = len(boxOfPermutationSpace) - len(agreementsRequired)
	permutationSpacesPerFolding: NDArray[numpy.integer] = (agreementsComplete.index.get_level_values('folding').value_counts()
		.reindex(range(len(arrayFoldings)), fill_value=0).add(permutationSpacesPinningNothing).to_numpy(dtype=numpy.int64))
	numpy.testing.assert_array_equal(permutationSpacesPerFolding, numpy.ones(len(arrayFoldings), dtype=numpy.int64)
		, err_msg=messageTestFailure(permutationSpacesPerFolding, 'one permutation space per folding', function, *arguments, **keywordArguments), strict=True)
