from astToolkit import extractFunctionDef, Make  # noqa: D100
from hunterMakesPy import raiseIfNone
from mapFolding.someAssemblyRequired import defaultA007822
import ast

ImaString: str = f"""
def {defaultA007822['function']['filterAsymmetricFolds']}({defaultA007822['variable']['stateInstance']}: MapFoldingState) -> MapFoldingState:
	{defaultA007822['variable']['stateInstance']}.indexLeaf = 0
	leafConnectee = 0
	while leafConnectee < {defaultA007822['variable']['stateInstance']}.leavesTotal + 1:
		leafNumber = int({defaultA007822['variable']['stateInstance']}.leafBelow[{defaultA007822['variable']['stateInstance']}.indexLeaf])
		{defaultA007822['variable']['stateInstance']}.leafComparison[leafConnectee] = (leafNumber - {defaultA007822['variable']['stateInstance']}.indexLeaf + {defaultA007822['variable']['stateInstance']}.leavesTotal) % {defaultA007822['variable']['stateInstance']}.leavesTotal
		{defaultA007822['variable']['stateInstance']}.indexLeaf = leafNumber
		leafConnectee += 1

	indexInMiddle = {defaultA007822['variable']['stateInstance']}.leavesTotal // 2
	{defaultA007822['variable']['stateInstance']}.indexMiniGap = 0
	while {defaultA007822['variable']['stateInstance']}.indexMiniGap < {defaultA007822['variable']['stateInstance']}.leavesTotal + 1:
		ImaSymmetricFold = True
		leafConnectee = 0
		while leafConnectee < indexInMiddle:
			if {defaultA007822['variable']['stateInstance']}.leafComparison[({defaultA007822['variable']['stateInstance']}.indexMiniGap + leafConnectee) % ({defaultA007822['variable']['stateInstance']}.leavesTotal + 1)] != {defaultA007822['variable']['stateInstance']}.leafComparison[({defaultA007822['variable']['stateInstance']}.indexMiniGap + {defaultA007822['variable']['stateInstance']}.leavesTotal - 1 - leafConnectee) % ({defaultA007822['variable']['stateInstance']}.leavesTotal + 1)]:
				ImaSymmetricFold = False
				break
			leafConnectee += 1
		{defaultA007822['variable']['stateInstance']}.{defaultA007822['variable']['counting']} += ImaSymmetricFold
		{defaultA007822['variable']['stateInstance']}.indexMiniGap += 1

	return {defaultA007822['variable']['stateInstance']}
"""

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
