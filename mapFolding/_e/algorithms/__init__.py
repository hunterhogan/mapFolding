"""Access experimental map-folding algorithms and crease validators.

You can count or collect folding representatives and check crease crossings. Supply either elimination
state or a map shape to the algorithm entry points, as documented by each module.

Modules
-------
constraintPropagation
	Enumerate representatives with a constraint solver.
elimination
	Reduce and branch permutation spaces to count representatives.
eliminationCrease
	Enumerate representatives with specialized crease reductions.
iff
	Check crease crossings and compute leaf-coordinate connections.
insertion
	Insert leaves incrementally and save a folding album.
insertion2上nDimensional吗
	Expand previous-dimension albums into constrained search spaces.
"""

from __future__ import annotations
