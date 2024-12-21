import jax

taskDivisions = jax.numpy.array(0, dtype=jax.numpy.int32)
"""Number of computation divisions"""
dimensionsTotal = jax.numpy.array(1, dtype=jax.numpy.int32)
"""Number of dimensions"""
leavesTotal = jax.numpy.array(2, dtype=jax.numpy.int32)
"""Total number of leaves"""

leafAbove = 0
"""Leaf above leaf m"""
leafBelow = 1
"""Leaf below leaf m"""
countDimensionsGapped = 2
"""Number of gaps available for leaf activeLeaf1ndex"""
gapRangeStart = 3
"""Index of gap stack for leaf activeLeaf1ndex"""

# leafAbove = jax.numpy.array(0, dtype=jax.numpy.int32)
# """Leaf above leaf m"""
# leafBelow = jax.numpy.array(1, dtype=jax.numpy.int32)
# """Leaf below leaf m"""
# countDimensionsGapped = jax.numpy.array(2, dtype=jax.numpy.int32)
# """Number of gaps available for leaf activeLeaf1ndex"""
# gapRangeStart = jax.numpy.array(3, dtype=jax.numpy.int32)
# """Index of gap stack for leaf activeLeaf1ndex"""

