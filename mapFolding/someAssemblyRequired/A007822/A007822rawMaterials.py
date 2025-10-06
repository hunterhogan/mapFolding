from astToolkit import extractFunctionDef, Make  # noqa: D100
from hunterMakesPy import raiseIfNone
from mapFolding.someAssemblyRequired import defaultA007822
import ast

ImaString: str = f"""
def {defaultA007822['function']['filterAsymmetricFolds']}({defaultA007822['variable']['stateInstance']}: {defaultA007822['variable']['stateDataclass']}) -> {defaultA007822['variable']['stateDataclass']}:
	{defaultA007822['variable']['stateInstance']}.indexLeaf = 1
	{defaultA007822['variable']['stateInstance']}.leafComparison[0] = 1
	{defaultA007822['variable']['stateInstance']}.leafConnectee = 1
	while {defaultA007822['variable']['stateInstance']}.leafConnectee < {defaultA007822['variable']['stateInstance']}.leavesTotal + 1:
		{defaultA007822['variable']['stateInstance']}.indexMiniGap = {defaultA007822['variable']['stateInstance']}.leafBelow[{defaultA007822['variable']['stateInstance']}.indexLeaf]
		{defaultA007822['variable']['stateInstance']}.leafComparison[{defaultA007822['variable']['stateInstance']}.leafConnectee] = ({defaultA007822['variable']['stateInstance']}.indexMiniGap - {defaultA007822['variable']['stateInstance']}.indexLeaf + {defaultA007822['variable']['stateInstance']}.leavesTotal) % {defaultA007822['variable']['stateInstance']}.leavesTotal
		{defaultA007822['variable']['stateInstance']}.indexLeaf = {defaultA007822['variable']['stateInstance']}.indexMiniGap
		{defaultA007822['variable']['stateInstance']}.leafConnectee += 1

	{defaultA007822['variable']['stateInstance']}.arrayGroupOfFolds = {defaultA007822['variable']['stateInstance']}.leafComparison[{defaultA007822['variable']['stateInstance']}.indicesArrayGroupOfFolds]
	{defaultA007822['variable']['stateInstance']}.{defaultA007822['variable']['counting']} += int(numpy.count_nonzero(numpy.all(numpy.equal({defaultA007822['variable']['stateInstance']}.arrayGroupOfFolds[..., slice(0, state.leavesTotal // 2)], {defaultA007822['variable']['stateInstance']}.arrayGroupOfFolds[..., slice(state.leavesTotal // 2, None)]), axis=1)))

	return {defaultA007822['variable']['stateInstance']}
"""

Import_numpy: ast.Import = Make.Import('numpy')

FunctionDef_filterAsymmetricFolds: ast.FunctionDef = raiseIfNone(extractFunctionDef(ast.parse(ImaString), defaultA007822['function']['filterAsymmetricFolds']))
del ImaString

ImaString = f"{defaultA007822['variable']['stateInstance']} = {defaultA007822['function']['filterAsymmetricFolds']}({defaultA007822['variable']['stateInstance']})"
A007822incrementCount = ast.parse(ImaString).body[0]
del ImaString

ImaString = f'{defaultA007822['variable']['stateInstance']}.{defaultA007822['variable']['counting']} = ({defaultA007822['variable']['stateInstance']}.{defaultA007822['variable']['counting']} + 1) // 2'
A007822adjustFoldsTotal = ast.parse(ImaString).body[0]
del ImaString

astExprCall_filterAsymmetricFoldsDataclass: ast.Expr = Make.Expr(Make.Call(Make.Name(defaultA007822['function']['filterAsymmetricFolds']), listParameters=[Make.Attribute(Make.Name(defaultA007822['variable']['stateInstance']), 'leafBelow')]))
astExprCall_filterAsymmetricFoldsLeafBelow: ast.Expr = Make.Expr(Make.Call(Make.Name(defaultA007822['function']['filterAsymmetricFolds']), listParameters=[Make.Name('leafBelow')]))
