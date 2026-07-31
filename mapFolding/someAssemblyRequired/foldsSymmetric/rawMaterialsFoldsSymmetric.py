# ruff:file-ignore[undocumented-public-module]
from __future__ import annotations

from astToolkit import extractFunctionDef, Make
from hunterMakesPy import raiseIfNone
from mapFolding.someAssemblyRequired import defaultFoldsSymmetric
from mapFolding.someAssemblyRequired.kitMakeModules import getModule
import ast

FunctionDef_filterAsymmetricFolds: ast.FunctionDef = raiseIfNone(extractFunctionDef(getModule(logicalPathInfix='algorithms', identifierModule='foldsSymmetric'), defaultFoldsSymmetric['function']['filterAsymmetricFolds']))

ImaString: str = f"{defaultFoldsSymmetric['variable']['stateInstance']} = {defaultFoldsSymmetric['function']['filterAsymmetricFolds']}({defaultFoldsSymmetric['variable']['stateInstance']})"
A007822incrementCount = ast.parse(ImaString).body[0]
del ImaString

ImaString = f'{defaultFoldsSymmetric['variable']['stateInstance']}.{defaultFoldsSymmetric['variable']['counting']} = ({defaultFoldsSymmetric['variable']['stateInstance']}.{defaultFoldsSymmetric['variable']['counting']} + 1) // 2'
A007822adjustFoldsTotal: ast.stmt = ast.parse(ImaString).body[0]
del ImaString

ExprCallFilterAsymmetricFolds_leafBelow: ast.Expr = Make.Expr(Make.Call(Make.Name(defaultFoldsSymmetric['function']['filterAsymmetricFolds']), listParameters=[Make.Name('leafBelow')]))
ExprCallFilterAsymmetricFoldsState: ast.Expr = Make.Expr(Make.Call(Make.Name(defaultFoldsSymmetric['function']['filterAsymmetricFolds']), listParameters=[Make.Name(defaultFoldsSymmetric['variable']['stateInstance'])]))
ExprCallFilterAsymmetricFoldsStateDot_leafBelow: ast.Expr = Make.Expr(Make.Call(Make.Name(defaultFoldsSymmetric['function']['filterAsymmetricFolds']), listParameters=[Make.Attribute(Make.Name(defaultFoldsSymmetric['variable']['stateInstance']), 'leafBelow')]))
