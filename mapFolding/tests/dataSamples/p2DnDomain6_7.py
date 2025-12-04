"""Verification data for combined leaf domains.

This module contains empirically extracted combined domain data for leaves
['6', '7'] across multiple mapShape configurations.

Each list is named `listDomain2D{dimensionsTotal}` where `dimensionsTotal`
is the exponent in the 2^dimensionsTotal mapShape, and it contains tuples representing
valid pile positions for the specified leaves. The tuple order follows the original
leaf argument order.
"""

listDomain2D4: list[tuple[int, ...]] = [(4, 5), (6, 3), (6, 7), (8, 9), (10, 3), (10, 7), (10, 11), (12, 5)]

listDomain2D5: list[tuple[int, ...]] = [(4, 5), (6, 3), (6, 7), (8, 9), (10, 3), (10, 7), (10, 11), (12, 5), (12, 13), (14, 3), (14, 7), (14, 11), (14, 15), (16, 17), (18, 3), (18, 7), (18, 15), (18, 19), (20, 5), (20, 13), (20, 21), (22, 7), (22, 11), (22, 23), (24, 9)]

listDomain2D6: list[tuple[int, ...]] = [(4, 5), (6, 3), (6, 7), (8, 9), (10, 3), (10, 7), (10, 11), (12, 5), (12, 13), (14, 3), (14, 7), (14, 11), (14, 15), (16, 17), (18, 3), (18, 7), (18, 11), (18, 15), (18, 19), (20, 5), (20, 13), (20, 21), (22, 3), (22, 7), (22, 11), (22, 15), (22, 19), (22, 23), (24, 9), (24, 25), (26, 3), (26, 7), (26, 15), (26, 23), (26, 27), (28, 5), (28, 13), (28, 21), (28, 29), (30, 3), (30, 7), (30, 11), (30, 19), (30, 27), (30, 31), (32, 33), (34, 3), (34, 7), (34, 11), (34, 19), (34, 23), (34, 31), (34, 35), (36, 5), (36, 13), (36, 29), (36, 37), (38, 7), (38, 27), (38, 39), (40, 9), (40, 25), (40, 41), (42, 11), (42, 15), (42, 23), (42, 43), (44, 13), (44, 21), (44, 45), (46, 15), (46, 19), (46, 47), (48, 17)]

