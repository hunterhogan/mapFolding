leafAbove = 0
"""Leaf above leaf m"""
leafBelow = 1
"""Leaf below leaf m"""
countDimensionsGapped = 2
"""Number of gaps available for leaf l"""
gapRangeStart = 3
"""Index of gap stack for leaf l"""

import jax

# taskDivisions = 0
# """Number of computation divisions"""
# taskIndex = 1
# """Index of computation division"""
# leavesTotal = 2
# """Total number of leaves"""
# dimensionsTotal = 3
# """Number of dimensions"""
taskDivisions = jax.numpy.int4(0)
"""Number of computation divisions"""
taskIndex = jax.numpy.int4(1)
"""Index of computation division"""
leavesTotal = jax.numpy.int4(2)
"""Total number of leaves"""
dimensionsTotal = jax.numpy.int4(3)
"""Number of dimensions"""
