"""Verification data for combined leaf domains.

This module contains empirically extracted combined domain data for leaves
['首零二', '首二'] across multiple mapShape configurations.

Each list is named `listDomain2D{dimensionsTotal}` where `dimensionsTotal`
is the exponent in the 2^dimensionsTotal mapShape, and it contains tuples representing
valid pile positions for the specified leaves. The tuple order follows the original
leaf argument order.
"""

listDomain2D5: list[tuple[int, ...]] = [(8, 7), (10, 9), (12, 11), (14, 13), (16, 15), (18, 13), (18, 17), (20, 11), (20, 19), (22, 9), (22, 17), (22, 21), (24, 7), (24, 23), (26, 9), (26, 13), (26, 17), (26, 21), (26, 25), (28, 11), (28, 19), (30, 13), (30, 17), (30, 21), (30, 25)]

listDomain2D6: list[tuple[int, ...]] = [(16, 15), (18, 17), (20, 19), (22, 21), (24, 23), (26, 25), (28, 27), (30, 29), (32, 31), (34, 29), (34, 33), (36, 27), (36, 35), (38, 25), (38, 33), (38, 37), (40, 23), (40, 39), (42, 21), (42, 29), (42, 37), (42, 41), (44, 19), (44, 35), (44, 43), (46, 17), (46, 29), (46, 33), (46, 41), (46, 45), (48, 15), (48, 47), (50, 17), (50, 21), (50, 37), (50, 41), (50, 45), (50, 49), (52, 19), (52, 27), (52, 35), (52, 43), (52, 51), (54, 21), (54, 29), (54, 33), (54, 41), (54, 45), (54, 49), (54, 53), (56, 23), (56, 39), (56, 55), (58, 25), (58, 29), (58, 33), (58, 37), (58, 41), (58, 45), (58, 49), (58, 53), (58, 57), (60, 27), (60, 35), (60, 43), (60, 51), (62, 29), (62, 33), (62, 37), (62, 41), (62, 45), (62, 49), (62, 53), (62, 57)]

