"""Compute possible pile domains for map leaves.

(AI generated docstring)

You can obtain shape-based pile ranges for one leaf or every leaf. You receive ranges independent of
the current pinning or reductions in an elimination state.

Contents
--------
Functions
	getDomainLeaf
		Compute the possible pile range for a leaf in the state's map.
	getLookupDomainsLeaves
		Build possible pile ranges keyed by leaf index.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding._e.dataBaskets import StateElimination
	from mapFolding.theTypes import Leaf

def getDomainLeaf(state: StateElimination, leaf: Leaf) -> range:
	"""Compute the possible pile range for a leaf in the state's map.

	(AI generated docstring)

	You can use the map shape in `state` to locate possible piles for `leaf`. You receive a restricted
	domain from the specialized implementation [1] when every dimension length is two, or every pile
	index for other shapes.

	Parameters
	----------
	state : StateElimination
		Elimination state providing the map shape and leaf count.
	leaf : Leaf
		Leaf index whose pile domain is requested.

	Returns
	-------
	domainLeaf : range
		Shape-based possible pile indices, independent of current pinning.

	See Also
	--------
	`getLookupDomainsLeaves`
		Compute the possible range for every leaf.

	References
	----------
	[1] `mapFolding._e.p2上nDimensional.leafDomains._getDomainLeaf`
	"""
	from mapFolding._e.p2上nDimensional.leafDomains import _getDomainLeaf  # ruff: ignore[import-outside-top-level]
	return _getDomainLeaf(leaf, state.totalDimensions, state.mapShape, state.totalLeaves)

def getLookupDomainsLeaves(state: StateElimination) -> dict[int, range]:
	"""Build possible pile ranges keyed by leaf index.

	You can compute `getDomainLeaf` [1] for every leaf in `state`. You receive a dictionary of
	shape-based domains without current pinning constraints.

	Parameters
	----------
	state : StateElimination
		Elimination state providing the shape and total number of leaves.

	Returns
	-------
	lookupDomainsLeaves : dict[int, range]
		Possible pile range for each leaf index.

	References
	----------
	[1] `getDomainLeaf`
	"""
	return {leaf: getDomainLeaf(state, leaf) for leaf in range(state.totalLeaves)}
