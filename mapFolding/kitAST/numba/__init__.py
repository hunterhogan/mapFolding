"""Access AST transformation modules for Numba compilation.

(AI generated docstring)

You can use this package to turn generated map-folding AST modules into Numba-compiled job
modules [1]. This package groups the code that applies `jit` decorators, stores reusable Numba
parameter sets, and writes specialized computation modules for specific map shapes. Proceed to
`mapFolding.kitAST.numba.kitNumba` [2] for decorator assembly and to
`mapFolding.kitAST.numba.makeJob` [3] for end-to-end job generation.

Modules
-------
kitNumba
	Apply Numba decorators and define reusable Numba compilation settings.
makeJob
	Generate specialized Numba job modules for specific map shapes.

References
----------
[1] Numba documentation.
	https://numba.readthedocs.io/en/stable/

[2] `mapFolding.kitAST.numba.kitNumba`

[3] `mapFolding.kitAST.numba.makeJob`
"""
