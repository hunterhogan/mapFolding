"""Obtain reusable syntax nodes for generating symmetric-fold modules.

(AI generated docstring)

You can use this module's Python abstract syntax tree nodes [1] when assembling synchronous and
asynchronous symmetric-fold modules. The module uses `astToolkit` [2] to extract
`mapFolding.algorithms.foldsSymmetric.filterAsymmetricFolds` [3] and create the call and assignment
nodes required by the module generators.

Contents
--------
Variables
    adjustTotalFolds
        Halve the generated symmetric-fold total with upward rounding.
    ExprCallFilterAsymmetricFolds_leafBelow
        Call the symmetry filter with the generated `leafBelow` name.
    ExprCallFilterAsymmetricFoldsState
        Call the symmetry filter with the generated state name.
    ExprCallFilterAsymmetricFoldsStateDot_leafBelow
        Call the symmetry filter with `leafBelow` from the generated state name.
    foldsSymmetricIncrementCount
        Replace a generated count increment with a symmetry-filter assignment.
    FunctionDef_filterAsymmetricFolds
        Insert the extracted symmetry-filter function into generated modules.

References
----------
[1] `ast` — Abstract syntax trees — Python documentation
    https://docs.python.org/3/library/ast.html
[2] astToolkit — Context7
    https://context7.com/hunterhogan/asttoolkit
[3] `mapFolding.algorithms.foldsSymmetric.filterAsymmetricFolds`

"""
from __future__ import annotations

from astToolkit import extractFunctionDef, Make
from hunterMakesPy import raiseIfNone
from mapFolding.kitAST.paths import getModule
from mapFolding.kitAST.theSSOT import defaultMapFoldingSymmetric
import ast

FunctionDef_filterAsymmetricFolds: ast.FunctionDef = raiseIfNone(extractFunctionDef(getModule(defaultMapFoldingSymmetric['module']['algorithmSource'], defaultMapFoldingSymmetric['logicalPath']['algorithm']), defaultMapFoldingSymmetric['function']['filterAsymmetricFolds']))
"""Insert the extracted symmetry-filter function into generated modules."""

ImaString: str = f"{defaultMapFoldingSymmetric['variable']['stateInstance']} = {defaultMapFoldingSymmetric['function']['filterAsymmetricFolds']}({defaultMapFoldingSymmetric['variable']['stateInstance']})"
foldsSymmetricIncrementCount: ast.stmt = ast.parse(ImaString).body[0]
"""Replace a generated count increment with a symmetry-filter assignment."""
del ImaString

ImaString = f'{defaultMapFoldingSymmetric['variable']['stateInstance']}.{defaultMapFoldingSymmetric['variable']['counting']} = ({defaultMapFoldingSymmetric['variable']['stateInstance']}.{defaultMapFoldingSymmetric['variable']['counting']} + 1) // 2'
adjustTotalFolds: ast.stmt = ast.parse(ImaString).body[0]
"""Halve the generated symmetric-fold total with upward rounding."""
del ImaString

ExprCallFilterAsymmetricFolds_leafBelow: ast.Expr = Make.Expr(Make.Call(Make.Name(defaultMapFoldingSymmetric['function']['filterAsymmetricFolds']), listParameters=[Make.Name('leafBelow')]))
"""Call the symmetry filter with the generated `leafBelow` name."""
ExprCallFilterAsymmetricFoldsState: ast.Expr = Make.Expr(Make.Call(Make.Name(defaultMapFoldingSymmetric['function']['filterAsymmetricFolds']), listParameters=[Make.Name(defaultMapFoldingSymmetric['variable']['stateInstance'])]))
"""Call the symmetry filter with the generated state name."""
ExprCallFilterAsymmetricFoldsStateDot_leafBelow: ast.Expr = Make.Expr(Make.Call(Make.Name(defaultMapFoldingSymmetric['function']['filterAsymmetricFolds']), listParameters=[Make.Attribute(Make.Name(defaultMapFoldingSymmetric['variable']['stateInstance']), 'leafBelow')]))
"""Call the symmetry filter with `leafBelow` from the generated state name."""
