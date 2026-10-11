"""Generate specialized computation modules from source syntax.

(AI generated docstring, and it's old)

You can use this package to derive counting variants, specialize jobs for individual map
shapes, and create lookup and documentation source for sequence formulas. The transformations
use `astToolkit` [1] to inspect and assemble Python syntax. The package exposes state-pattern
predicates through `IfThis` [2] and datatype-import configuration through `Settings形` [3].

Modules
-------
dataclasses
	Decompose state dataclasses and adapt field-based calls.
linux
	Prepare source, compile Linux jobs, and strip binaries.
oeisIDByFormula
	Generate known-value lookup variants of formula source.
paths
	Resolve source module identifiers and output filenames.
prefab
	Apply common syntax edits and write generated modules.
RecipeJob
	Configure and specialize counting jobs.
theSSOT
	Supply shared identifiers, destinations, and count estimates.

Subpackages
-----------
codon
	Write jobs and compile source with Codon on Linux.
mapFolding
	Generate multidimensional counting variants.
mapFoldingSymmetric
	Generate synchronous and asynchronous symmetric variants.
matrixMeanders
	Generate matrix-meander counting variants.
numba
	Assemble compiler settings, decorators, and counting jobs.
zCuzDocStoopid
	Generate documentation from sequence metadata.

Exports
-------
IfThis
	Build comparison and control-flow predicates.
Settings形
	Configure datatype import substitutions.

References
----------
[1] astToolkit tree visitors and source containers.
	https://context7.com/hunterhogan/asttoolkit
[2] `mapFolding.kitAST._toolIfThis.IfThis`

[3] `mapFolding.kitAST._containers.Settings形`
"""

# isort: split
from __future__ import annotations

# isort: split
from mapFolding.kitAST._toolIfThis import IfThis as IfThis

# isort: split
from mapFolding.kitAST._containers import Settings形 as Settings形

# TODO Problem: `LedgerOfImports` from `astToolkit` does not handle TYPE_CHECKING blocks very well.
