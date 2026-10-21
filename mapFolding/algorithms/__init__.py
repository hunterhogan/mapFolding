"""Hand-made algorithms and formulas.

Modules
-------
constraintPropagation
	Enumerate folding representatives with a constraint solver.
daoOfMapFolding
	Count multidimensional map foldings with insertion search.
elimination
	Count folding representatives by reducing and branching permutation spaces.
eliminationCrease
	Enumerate foldings of maps whose dimension lengths are all two.
foldsSymmetric
	Count rotationally symmetric stamp foldings.
iff
	Check leaf arrangements for forbidden crease crossings.
insertion
	Generate and save arch-diagram representatives by inserting endpoints.
insertion2上nDimensional
	Expand smaller-map albums into constrained higher-dimensional search spaces.
matrixMeanders
	Count meanders with dictionary-based transfer states.
matrixMeandersNumPy
	Count meanders with NumPy array storage.
matrixMeandersPandas
	Count meanders with Pandas table storage.
matrixMeandersShare
	Share arch-code operations and integer-width checks between meander algorithms.
permutations
	Generate stamp-folding and meander permutations.
permutationsBilateral
	Count bilateral stamp-folding and meander arrangements.
permutationsBilateralConcurrent
	Count bilateral arrangements with concurrent tasks.
Y7Y_catalanArch
	Enumerate arch codes and count successive endpoint removals.
"""

from __future__ import annotations
