"""Triangles of integers keyed by row number, and a sequence whose consecutive values form a triangle.

Cell values are primes, so a misplaced cell names its origin. Consumers do not mutate these objects.
"""
from __future__ import annotations

triangleDense: dict[int, tuple[int, ...]] = {2: (11, 13), 3: (17, 19, 23), 5: (29,)}
"""Row 4 is absent, so consumers meet a gap between rows 3 and 5."""
triangleHalf: dict[int, tuple[int, ...]] = {2: (503,), 3: (509,), 4: (521, 523), 5: (541,)}
"""Each row n has floor(n/2) cells, the layout of the semi-meander triangle."""
triangleInt64Edges: dict[int, tuple[int, ...]] = {2: (-9223372036854775808, 9223372036854775807), 3: (0, -1, 9007199254740991)}
triangleBeyondInt64: dict[int, tuple[int, ...]] = {2: (-9223372036854775809, 9223372036854775808), 3: (9007199254740992, 9007199254740993, 2 ** 100)}

lookupTriangles: dict[str, dict[int, tuple[int, ...]]] = {
	'triangleDense': triangleDense,
	'triangleHalf': triangleHalf,
	'triangleInt64Edges': triangleInt64Edges,
	'triangleBeyondInt64': triangleBeyondInt64,
}

sequenceHalf: dict[int, int] = {2: 601, 3: 607, 4: 613, 5: 617, 6: 619}
"""Consecutive values from n = 2 in the layout of `triangleHalf`: rows 2, 3, and 4 are complete, and row 5 is partial."""
