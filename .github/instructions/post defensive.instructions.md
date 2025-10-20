---
applyTo: 'mapFolding/algorithms/*.py'
---

These modules are called *after* all defensive checks.

Avoid early-return guard clauses, short-circuit returns, and multiple exit points. This codebase enforces a
single-return-per-function pattern with stable shapes/dtypes due to AST transforms. An empty input is a problem, so allow it to
fail early.

If an algorithm has potential for infinite loops, fix the root cause: do NOT add artificial safety limits (e.g., maxIterations
counters) to prevent infinite loops.

Do not guard vectorized or masked operations behind data-dependent boolean reductions. Never write conditionals like
`if numpy.any(mask):` or `if numpy.all(mask):` (or the moral equivalent using `count_nonzero`). Apply the masked operation
directly; empty masks naturally no-op. This keeps control flow stable for AST transforms and avoids data-dependent branching.

Always use semantic column, index, or slice identifiers: Never hardcode the locations.
