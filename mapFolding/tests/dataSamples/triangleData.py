from __future__ import annotations

triangleDense: dict[int, tuple[int, ...]] = {2: (11, 13), 3: (17, 19, 23), 5: (29,)}
triangleSparse: dict[int, dict[int, int]] = {5: {5: 113}, 3: {3: 109, 1: 107}, 2: {2: 103, 1: 101}}
triangleGaps: dict[int, dict[int, int]] = {3: {3: 229, 1: 227}, 1: {1: 211}, 2: {2: 223}}
sequenceGaps: dict[int, int] = {6: 421, 1: 401, 4: 419, 3: 409}
triangleHalf: dict[int, tuple[int, ...]] = {2: (503,), 3: (509,), 4: (521, 523), 5: (541,)}
sequenceHalf: dict[int, int] = {2: 601, 3: 607, 4: 613, 5: 617, 6: 619}
contentsBFile: str = '# prime observations\n2 701\n3 709\n'

triangleInt64Edges: dict[int, tuple[int, ...]] = {2: (-9223372036854775808, 9223372036854775807), 3: (0, -1, 9007199254740991)}
triangleBeyondInt64: dict[int, tuple[int, ...]] = {2: (-9223372036854775809, 9223372036854775808), 3: (9007199254740992, 9007199254740993, 2 ** 100)}
triangleSparseEdges: dict[int, dict[int, int]] = {3: {3: 2 ** 1024, 1: -(2 ** 1024)}, 2: {2: 18446744073709551616, 1: 18446744073709551615}}
