"""Assert contracts about mapFolding's own domain objects, which `hunterMakesPy.tests` cannot depend on.

The shared assertion helpers come from `hunterMakesPy.tests` through `mapFolding.tests`.
"""
from __future__ import annotations

from hunterMakesPy.tests import messageTestFailure
from mapFolding.filters import makeAgreements
from typing import TYPE_CHECKING
import numpy
import numpy.testing

if TYPE_CHECKING:
	from collections.abc import Sequence
	from mapFolding.dataBaskets import PermutationSpace
	from numpy.typing import NDArray
	from typing import Any

def assertPermutationSpacesPartitionFoldings(boxOfPermutationSpace: Sequence[PermutationSpace], arrayFoldings: NDArray[numpy.uint8]
	, function: str, *arguments: Any, **keywordArguments: Any) -> None:
	"""Assert that every folding agrees with exactly one permutation space, as `mapFolding.filters.makeAgreements` defines agreement."""
	permutationSpacesPerFolding: NDArray[numpy.int64] = makeAgreements(boxOfPermutationSpace, arrayFoldings).sum(axis=1, dtype=numpy.int64)
	numpy.testing.assert_array_equal(permutationSpacesPerFolding, numpy.ones(len(arrayFoldings), dtype=numpy.int64)
		, err_msg=messageTestFailure(permutationSpacesPerFolding, 'one permutation space per folding', function, *arguments, **keywordArguments), strict=True)
