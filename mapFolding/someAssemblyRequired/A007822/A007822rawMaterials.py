from astToolkit import extractFunctionDef, Make  # noqa: D100
from hunterMakesPy import raiseIfNone
from mapFolding.someAssemblyRequired import defaultA007822
from mapFolding.someAssemblyRequired.toolkitMakeModules import getModule
import ast

FunctionDef_filterAsymmetricFolds: ast.FunctionDef = raiseIfNone(extractFunctionDef(getModule(logicalPathInfix='algorithms', identifierModule='symmetricFolds'), defaultA007822['function']['filterAsymmetricFolds']))

ImaString: str = f"{defaultA007822['variable']['stateInstance']} = {defaultA007822['function']['filterAsymmetricFolds']}({defaultA007822['variable']['stateInstance']})"
A007822incrementCount = ast.parse(ImaString).body[0]
del ImaString

ImaString = f'{defaultA007822['variable']['stateInstance']}.{defaultA007822['variable']['counting']} = ({defaultA007822['variable']['stateInstance']}.{defaultA007822['variable']['counting']} + 1) // 2'
A007822adjustFoldsTotal: ast.stmt = ast.parse(ImaString).body[0]
del ImaString

ExprCallFilterAsymmetricFolds_leafBelow: ast.Expr = Make.Expr(Make.Call(Make.Name(defaultA007822['function']['filterAsymmetricFolds']), listParameters=[Make.Name('leafBelow')]))
ExprCallFilterAsymmetricFoldsState: ast.Expr = Make.Expr(Make.Call(Make.Name(defaultA007822['function']['filterAsymmetricFolds']), listParameters=[Make.Name(defaultA007822['variable']['stateInstance'])]))
ExprCallFilterAsymmetricFoldsStateDot_leafBelow: ast.Expr = Make.Expr(Make.Call(Make.Name(defaultA007822['function']['filterAsymmetricFolds']), listParameters=[Make.Attribute(Make.Name(defaultA007822['variable']['stateInstance']), 'leafBelow')]))
