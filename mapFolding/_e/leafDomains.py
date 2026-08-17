"""Describe valid pile domains for each leaf.

(AI generated docstring)

You can use this module to compute the Python `range` of candidate `Pile` values for a
single `Leaf` or for every `Leaf` in an elimination state.

Contents
--------
Functions
	getDomainLeaf
		Compute the candidate `Pile` range for one `leaf`.
	getLookupDomainsLeaves
		Build the candidate `Pile` range for each `leaf`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding._e.dataBaskets import EliminationState
	from mapFolding._e.theTypes import Leaf

def getDomainLeaf(state: EliminationState, leaf: Leaf) -> range:
	"""Compute the candidate `Pile` range for one `leaf`.

	(AI generated docstring)

	You can use `getDomainLeaf` to retrieve the Python `range` of `Pile` values where
	`leaf` may appear for the map shape described by `state`.

	Parameters
	----------
	state : EliminationState
		Elimination state that provides the map dimensions and total `Leaf` count.
	leaf : Leaf
		`Leaf` index whose candidate `Pile` range is requested.

	Returns
	-------
	domainLeaf : range
		Python `range` of candidate `Pile` values for `leaf`.

	See Also
	--------
	`getLookupDomainsLeaves`
		Build the candidate `Pile` range for each `leaf` in `state`.
	"""
	from mapFolding._e._2上nDimensional.leafDomains import _getDomainLeaf  # ruff: ignore[import-outside-top-level]
	return _getDomainLeaf(leaf, state.totalDimensions, state.mapShape, state.totalLeaves)

def getLookupDomainsLeaves(state: EliminationState) -> dict[int, range]:
	"""Dictionary of `Leaf` to `range` of `Pile` in which `Leaf` may be found in a `Folding`.

	For each `Leaf`, the associated Python `range` defines the mathematical domain:
	1. every `Pile` at which `Leaf` may be found in a `Folding` and
	2. in the set of all valid `Folding`, every `Pile` at which `Leaf` must be found.
	"""
	return {leaf: getDomainLeaf(state, leaf) for leaf in range(state.totalLeaves)}
