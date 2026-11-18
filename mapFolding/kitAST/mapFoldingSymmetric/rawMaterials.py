"""Access reusable filter-call and count-adjustment syntax.

(AI generated docstring)

You can use this module to obtain reusable Python syntax nodes. Importing the module
loads and extracts `mapFolding.algorithms.foldsSymmetric.filterAsymmetricFolds`
and creates call and assignment nodes using the configured state identifiers.

Contents
--------
Variables
	adjustTotalFolds
		Access a statement that halves the symmetric count with upward rounding.
	ExprCallFilterAsymmetricFolds_leafBelow
		Access a filter call with the generated `leafBelow` name.
	ExprCallFilterAsymmetricFoldsState
		Access a filter call with the configured state name.
	ExprCallFilterAsymmetricFoldsStateDot_leafBelow
		Access a filter call with the state `leafBelow` attribute.
	foldsSymmetricIncrementCount
		Access an assignment that updates state through the symmetry filter.
	FunctionDef_filterAsymmetricFolds
		Access the extracted symmetry-filter function definition.
"""
from __future__ import annotations

from astToolkit import Make
from astToolkit.changeDef import extractFunctionDef
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
