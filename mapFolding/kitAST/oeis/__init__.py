"""Generate OEIS formula modules, their tests, and their reference samples.

(AI generated docstring)

You can use this package to regenerate `mapFolding.oeis.byFormula` and the tests of its formulas
from the symbolic relations in `mapFolding.oeis`. The transformations use `astToolkit` to build
Python syntax.

Modules
-------
oeisIDByFormula
	Generate sequence-formula source from the symbolic relations.
oeisModules
	Write the formula module, its tests, and its reference samples.
oeisTests
	Generate the tests of the formulas and of the triangle relations, and their reference samples.
"""

from __future__ import annotations
