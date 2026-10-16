from __future__ import annotations

triangleDense: dict[int, tuple[int, ...]] = {2: (11, 13), 3: (17, 19, 23), 5: (29,)}
triangleSparse: dict[int, dict[int, int]] = {2: {1: 11, 2: 13}, 3: {1: 17, 3: 23}, 5: {5: 31}}
triangleGaps: dict[int, dict[int, int]] = {1: {1: 11}, 2: {2: 17}, 3: {1: 19, 3: 29}}
sequenceGaps: dict[int, int] = {1: 11, 3: 17, 4: 19, 6: 29}
triangleHalf: dict[int, tuple[int, ...]] = {2: (11,), 3: (13,), 4: (17, 19), 5: (23,)}
sequenceHalf: dict[int, int] = {2: 11, 3: 13, 4: 17, 5: 19, 6: 23}
contentsBFile: str = '# prime observations\n2 11\n3 13\n'
