"""
Inspect archived implementations of Lunnon's map-folding search.

You can use this package to compare static archived modules that all implement the same backtracking
search for distinct multi-dimensional map foldings [1]. The archive is organized around changes in
state naming, array representation, control-flow presentation, compilation strategy, and publication
of large computed results, including the first completed 2×19 and 2×20 values for OEIS A001415 [2].

Modules
-------
flattened
   Decomposes the backtracking loop into named predicates and state transitions. The structure is
   expository, not a helper-function layer.
hunterNumba
   Manually specializes array widths and compiles the hot loop with Numba [3].
irvineJavaPort
   Ports Sean A. Irvine's Java implementation [4] and preserves its residue/modulus work-partition
   interface.
jaxCount
   Recasts the mutable search state as tuple-carried JAX control flow [5] for accelerator experiments.
lunnonNumpy
   Keeps Lunnon's search recognizable while storing the core arrays as NumPy arrays [6].
lunnonWhile
   Keeps the search in explicit Python `while` form, which makes comparison against the Atlas Autocode
   transcript [7] straightforward.
rotatedEntryPoint
   Re-enters the same search state machine from a rotated control-flow position to test equivalent
   organization.
total_countPlus1vsPlusN
   Isolates the performance effect of counting completed foldings by `+1` versus batched `+n`.

Subpackages
-----------
jobsCompleted
   Records source files, saved states, and results for the first completed 2×19 and 2×20 computations.

Shared algorithm
----------------
Each variant computes cumulative products, derives per-dimension leaf coordinates, builds a
leaf-connection relation, enumerates admissible gaps for each active leaf, and backtracks through
insertions until every leaf has been placed. The variants differ less in mathematical result than in
how the search state is named, stored, compiled, entered, or counted.

Reference materials
-------------------
foldings.AA
   Reconstructed Atlas Autocode listing with corrected paper-era typographical faults.
foldings.txt
   Transcript of the published algorithm listing from Lunnon's paper [1].
notes.md
   Working terminology and identifier alignments across the archived variants.
Speed highlights.md
   Timing chronology for major optimization milestones inside the archive.

Archive status
--------------
This archive is valuable for historical comparison, correctness cross-checks, and targeted
optimization study, but this archive is not the maintained production path of the repository.

References
----------
[1] W. F. Lunnon. "Multi-dimensional map-folding". The Computer Journal,
   Volume 14, Issue 1, 1971, pp. 75-80. https://doi.org/10.1093/comjnl/14.1.75
[2] OEIS A001415. Number of ways of folding a 2 X n strip of stamps.
   https://oeis.org/A001415
[3] Numba documentation.
   https://numba.readthedocs.io/en/stable/
[4] Sean A. Irvine. `A001415.java`.
   https://github.com/archmageirvine/joeis/blob/80e3e844b11f149704acbab520bc3a3a25ac34ff/src/irvine/oeis/a001/A001415.java
[5] JAX documentation.
   https://docs.jax.dev/en/latest/
[6] NumPy reference.
   https://numpy.org/doc/stable/reference/index.html
[7] Atlas Autocode - Wikipedia.
   https://en.wikipedia.org/wiki/Atlas_Autocode
"""
