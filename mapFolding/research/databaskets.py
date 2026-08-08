# TODO https://github.com/python/typing/discussions/2092

from __future__ import annotations

from mapFolding import Array1DLeavesTotal
from mapFolding._e import leafOrigin
from mapFolding.beDRY import makeDataContainer
from mapFolding.dataBaskets import MapFoldingState
from mapFolding.oeis import getFoldsTotalKnown
import dataclasses
import numpy
import typing

def _extract_numpy_dtype(t: typing.Any) -> typing.Any:
	if isinstance(t, type) and issubclass(t, numpy.generic):
		return typing.cast("typing.Any", t)
	for arg in typing.get_args(t):
		try:
			res = _extract_numpy_dtype(arg)
			if res is not None:
				return res
		except TypeError:
			pass
	message: str = f"No numpy.generic type found in {t}"
	raise TypeError(message)


@dataclasses.dataclass(slots=True)
class LeafSequenceState(MapFoldingState):
	"""Specialized computational state for tracking leaf sequences during analysis.

	(AI generated docstring)

	This class extends the base MapFoldingState with additional capability for recording and analyzing
	the sequence of leaf connections discovered during map folding computations. It integrates with
	the OEIS (Online Encyclopedia of Integer Sequences) system to leverage known sequence data for
	optimization and validation.

	The leaf sequence tracking is particularly valuable for research and verification purposes,
	allowing detailed analysis of how folding patterns emerge and enabling comparison with established
	mathematical sequences.

	Attributes
	----------
	leafSequence : Array1DLeavesTotal = None
		Array storing the sequence of leaf connections discovered.
	"""

	leafSequence: Array1DLeavesTotal = dataclasses.field(default=None, init=True)
	"""
	Array storing the sequence of leaf connections discovered during computation.

	This array records the order in which leaf connections are established during the folding
	analysis. The sequence provides insights into the algorithmic progression and can be compared
	against known mathematical sequences for validation and optimization purposes.
	"""

	def __post_init__(self) -> None:
		"""Initialize sequence tracking arrays with OEIS integration.

		(AI generated docstring)

		This method performs base initialization then sets up the leaf sequence tracking array. It
		queries the OEIS system for known fold totals corresponding to the current map shape, using
		this information to optimally size the sequence tracking array.

		Notes
		-----
		The sequence array is automatically initialized to record the starting leaf connection,
		providing a foundation for subsequent sequence tracking.

		"""
		super().__post_init__()
		if self.leafSequence is None:
			foldsTotalKnown: int | None = getFoldsTotalKnown(self.mapShape)
			if foldsTotalKnown is not None:
				groupsOfFoldsKnown: int = foldsTotalKnown // self.leavesTotal
				hints = typing.get_type_hints(self.__class__)
				dtype = _extract_numpy_dtype(hints['leafSequence'])
				self.leafSequence = makeDataContainer(groupsOfFoldsKnown, dtype)
				# I previously collected a lot of data using `Leaf` numbers from 1 to `leavesTotal`,
				# so I initialized the array with `self.leaf1ndex` instead of `leafOrigin`.
				self.leafSequence[self.groupsOfFolds] = leafOrigin
