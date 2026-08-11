# DOCUMENT
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding._e.dataBaskets import EliminationState
	from mapFolding._e.theTypes import Leaf

def getDomainLeaf(state: EliminationState, leaf: Leaf) -> range:
	# DOCUMENT
	# TODO Is this the "right" way to do this given that I want to segregate `_2上nDimensional`?
	from mapFolding._e._2上nDimensional.leafDomains import _getDomainLeaf  # ruff: ignore[import-outside-top-level]
	return _getDomainLeaf(leaf, state.totalDimensions, state.mapShape, state.totalLeaves)

def getLookupDomainsLeaves(state: EliminationState) -> dict[int, range]:
	"""Dictionary of `Leaf` to `range` of `Pile` in which `Leaf` may be found in a `Folding`.

	For each `Leaf`, the associated Python `range` defines the mathematical domain:
	1. every `Pile` at which `Leaf` may be found in a `Folding` and
	2. in the set of all valid `Folding`, every `Pile` at which `Leaf` must be found.
	"""
	return {leaf: getDomainLeaf(state, leaf) for leaf in range(state.totalLeaves)}
