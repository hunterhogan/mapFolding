"""Verification data for combined leaf domains.

This module contains empirically extracted combined domain data for leaves
['5', '4'] across multiple mapShape configurations.

Each list is named `list2D{dimensionsTotal}Domain5_4` where `dimensionsTotal`
is the exponent in the 2^dimensionsTotal mapShape, and it contains tuples representing
valid pile positions for the specified leaves. The tuple order follows the original
leaf argument order.
"""

list2D4Domain5_4: list[tuple[int, ...]] = [(2, 7), (2, 11), (4, 13), (6, 7), (6, 11), (8, 9), (10, 11), (12, 13)]

list2D5Domain5_4: list[tuple[int, ...]] = [(2, 7), (2, 11), (2, 15), (2, 19), (4, 13), (4, 21), (6, 7), (6, 11), (6, 15), (6, 19), (6, 23), (8, 9), (8, 25), (10, 11), (10, 15), (10, 23), (12, 13), (12, 21), (14, 15), (14, 19), (16, 17), (18, 19), (20, 21), (22, 23), (24, 25)]

list2D6Domain5_4: list[tuple[int, ...]] = [(2, 7), (2, 11), (2, 15), (2, 19), (2, 23), (2, 27), (2, 31), (2, 35), (4, 13), (4, 21), (4, 29), (4, 37), (6, 7), (6, 11), (6, 15), (6, 19), (6, 23), (6, 27), (6, 31), (6, 35), (6, 39), (8, 9), (8, 25), (8, 41), (10, 11), (10, 15), (10, 19), (10, 23), (10, 31), (10, 35), (10, 43), (12, 13), (12, 21), (12, 29), (12, 37), (12, 45), (14, 15), (14, 19), (14, 23), (14, 27), (14, 43), (14, 47), (16, 17), (16, 49), (18, 19), (18, 23), (18, 31), (18, 35), (18, 47), (20, 21), (20, 29), (20, 45), (22, 23), (22, 27), (22, 35), (22, 43), (24, 25), (24, 41), (26, 27), (26, 31), (26, 39), (28, 29), (28, 37), (30, 31), (30, 35), (32, 33), (34, 35), (36, 37), (38, 39), (40, 41), (42, 43), (44, 45), (46, 47), (48, 49)]

