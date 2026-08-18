# ruff: ignore[undocumented-public-module]
# FIXME https://github.com/python/typing/discussions/2092

from __future__ import annotations

from mapFolding._e import leafOrigin
from mapFolding.beDRY import makeDataContainer
from mapFolding.dataBaskets import StateMapFolding
from mapFolding.oeis import getTotalFoldsKnown
from mapFolding.theTypes import 形Array1DTotalLeaves
import dataclasses

@dataclasses.dataclass(slots=True)
class LeafSequenceState(StateMapFolding):
	"""Specialized computational state for tracking leaf sequences during analysis.

	(AI generated docstring)

	This class extends the base StateMapFolding with additional capability for recording and analyzing
	the sequence of leaf connections discovered during map folding computations. It integrates with
	the OEIS (Online Encyclopedia of Integer Sequences) system to leverage known sequence data for
	optimization and validation.

	The leaf sequence tracking is particularly valuable for research and verification purposes,
	allowing detailed analysis of how folding patterns emerge and enabling comparison with established
	mathematical sequences.

	Attributes
	----------
	leafSequence : 形Array1DTotalLeaves = None
		Array storing the sequence of leaf connections discovered.
	"""

	leafSequence: 形Array1DTotalLeaves = dataclasses.field(default_factory=lambda: 形Array1DTotalLeaves([]), init=True, metadata={'dtype': 形Array1DTotalLeaves.__args__[1].__args__[0]})
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
		if not self.leafSequence.shape:
			totalFoldsKnown: int | None = getTotalFoldsKnown(self.mapShape)
			if totalFoldsKnown is not None:
				groupsOfFoldsKnown: int = totalFoldsKnown // self.totalLeaves
				self.leafSequence = makeDataContainer(groupsOfFoldsKnown, self.__dataclass_fields__['leafSequence'].metadata['dtype'])
				# I previously collected a lot of data using `Leaf` numbers from 1 to `totalLeaves`,
				# so I initialized the array with `self.leaf1ndex` instead of `leafOrigin`.
				self.leafSequence[self.groupsOfFolds] = leafOrigin
