"""Verification data for combined leaf domains.

This module contains empirically extracted combined domain data for leaves
['7', '6'] across multiple mapShape configurations.

Each list is named `listDomain2D{dimensionsTotal}` where `dimensionsTotal`
is the exponent in the 2^dimensionsTotal mapShape, and it contains tuples representing
valid pile positions for the specified leaves. The tuple order follows the original
leaf argument order.
"""

listDomain2D4: list[tuple[int, ...]] = [(3, 6), (3, 10), (5, 4), (5, 12), (7, 6), (7, 10), (9, 8), (11, 10)]

listDomain2D5: list[tuple[int, ...]] = [(3, 6), (3, 10), (3, 14), (3, 18), (5, 4), (5, 12), (5, 20), (7, 6), (7, 10), (7, 14), (7, 18), (7, 22), (9, 8), (9, 24), (11, 10), (11, 14), (11, 22), (13, 12), (13, 20), (15, 14), (15, 18), (17, 16), (19, 18), (21, 20), (23, 22)]

listDomain2D6: list[tuple[int, ...]] = [(3, 6), (3, 10), (3, 14), (3, 18), (3, 22), (3, 26), (3, 30), (3, 34), (5, 4), (5, 12), (5, 20), (5, 28), (5, 36), (7, 6), (7, 10), (7, 14), (7, 18), (7, 22), (7, 26), (7, 30), (7, 34), (7, 38), (9, 8), (9, 24), (9, 40), (11, 10), (11, 14), (11, 18), (11, 22), (11, 30), (11, 34), (11, 42), (13, 12), (13, 20), (13, 28), (13, 36), (13, 44), (15, 14), (15, 18), (15, 22), (15, 26), (15, 42), (15, 46), (17, 16), (17, 48), (19, 18), (19, 22), (19, 30), (19, 34), (19, 46), (21, 20), (21, 28), (21, 44), (23, 22), (23, 26), (23, 34), (23, 42), (25, 24), (25, 40), (27, 26), (27, 30), (27, 38), (29, 28), (29, 36), (31, 30), (31, 34), (33, 32), (35, 34), (37, 36), (39, 38), (41, 40), (43, 42), (45, 44), (47, 46)]

