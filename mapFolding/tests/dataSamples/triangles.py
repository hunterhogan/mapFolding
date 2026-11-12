"""Triangles of integers keyed by row number, and a sequence whose consecutive values form a triangle.

Cell values are primes, so a misplaced cell names its origin. Consumers do not mutate these objects.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from _pytest.mark.structures import ParameterSet

# Row 4 is absent, so consumers meet a gap between rows 3 and 5.
triangleDense: dict[int, tuple[int, ...]] = {2: (11, 13), 3: (17, 19, 23), 5: (29,)}
# Each row n has floor(n/2) cells, the layout of the semi-meander triangle.
triangleHalf: dict[int, tuple[int, ...]] = {2: (503,), 3: (509,), 4: (521, 523), 5: (541,)}
triangleInt64Edges: dict[int, tuple[int, ...]] = {2: (-9223372036854775808, 9223372036854775807), 3: (0, -1, 9007199254740991)}
triangleBeyondInt64: dict[int, tuple[int, ...]] = {2: (-9223372036854775809, 9223372036854775808), 3: (9007199254740992, 9007199254740993, 2 ** 100)}

boxOfTriangles: tuple[ParameterSet, ...] = (
	pytest.param(triangleDense, id='triangleDense')
	, pytest.param(triangleHalf, id='triangleHalf')
	, pytest.param(triangleInt64Edges, id='triangleInt64Edges')
	, pytest.param(triangleBeyondInt64, id='triangleBeyondInt64')
)

# Consecutive values from n = 2 in the layout of `triangleHalf`: rows 2, 3, and 4 are complete, and row 5 is partial.
sequenceHalf: dict[int, int] = {2: 601, 3: 607, 4: 613, 5: 617, 6: 619}
