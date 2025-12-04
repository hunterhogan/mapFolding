"""Verification data for combined leaf domains.

This module contains empirically extracted combined domain data for leaves
['首零一二', '首一二'] across multiple mapShape configurations.

Each list is named `listDomain2D{dimensionsTotal}` where `dimensionsTotal`
is the exponent in the 2^dimensionsTotal mapShape, and it contains tuples representing
valid pile positions for the specified leaves. The tuple order follows the original
leaf argument order.
"""

listDomain2D5: list[tuple[int, ...]] = [(9, 10), (11, 12), (13, 14), (15, 16), (17, 14), (17, 18), (19, 12), (19, 20), (21, 10), (21, 18), (21, 22), (23, 8), (23, 24), (25, 10), (25, 14), (25, 18), (25, 22), (25, 26), (27, 12), (27, 20), (27, 28), (29, 14), (29, 18), (29, 22), (29, 26)]

listDomain2D6: list[tuple[int, ...]] = [(17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32), (33, 30), (33, 34), (35, 28), (35, 36), (37, 26), (37, 34), (37, 38), (39, 24), (39, 40), (41, 22), (41, 30), (41, 38), (41, 42), (43, 20), (43, 36), (43, 44), (45, 18), (45, 30), (45, 34), (45, 42), (45, 46), (47, 16), (47, 48), (49, 18), (49, 22), (49, 38), (49, 42), (49, 46), (49, 50), (51, 20), (51, 28), (51, 36), (51, 44), (51, 52), (53, 22), (53, 30), (53, 34), (53, 42), (53, 46), (53, 50), (53, 54), (55, 24), (55, 40), (55, 56), (57, 26), (57, 30), (57, 34), (57, 38), (57, 42), (57, 46), (57, 50), (57, 54), (57, 58), (59, 28), (59, 36), (59, 44), (59, 52), (59, 60), (61, 30), (61, 34), (61, 38), (61, 42), (61, 46), (61, 50), (61, 54), (61, 58)]

