from astToolkit import extractFunctionDef, Make  # noqa: D100
from hunterMakesPy import raiseIfNone
from mapFolding.someAssemblyRequired import defaultA007822
import ast

ImaString = f"""
def getListIndicesSymmetricInGroupOfFolds({defaultA007822['variable']['stateInstance']}: {defaultA007822['variable']['stateDataclass']}) -> list[list[tuple[int, int]]]:
	indexAtMidpoint: int = {defaultA007822['variable']['stateInstance']}.leavesTotal // 2
	normalizedRangeLeft: list[int] = [*range({defaultA007822['variable']['stateInstance']}.leavesTotal + 1), *range((({defaultA007822['variable']['stateInstance']}.leavesTotal + 1)// 2) - 1)]
	normalizedRangeRight: list[int] = [*range({defaultA007822['variable']['stateInstance']}.leavesTotal - 2, -1, -1), *range({defaultA007822['variable']['stateInstance']}.leavesTotal, indexAtMidpoint - 1, -1)]
	listIndicesSymmetricInGroupOfFolds: list[list[tuple[int, int]]] = []
	for group in range({defaultA007822['variable']['stateInstance']}.leavesTotal + 1):
		normalIndicesLeft: list[int] = normalizedRangeLeft[group:group+indexAtMidpoint]
		normalIndicesRight: list[int] = normalizedRangeRight[{defaultA007822['variable']['stateInstance']}.leavesTotal-group:{defaultA007822['variable']['stateInstance']}.leavesTotal-group+indexAtMidpoint]
		listIndicesSymmetric: list[tuple[int, int]] = [(indexLeft, indexRight) for indexLeft, indexRight in zip(normalIndicesLeft, normalIndicesRight, strict=True)]
		listIndicesSymmetricInGroupOfFolds.append(listIndicesSymmetric)
	return listIndicesSymmetricInGroupOfFolds
"""
FunctionDefGetIndices: ast.FunctionDef = raiseIfNone(extractFunctionDef(ast.parse(ImaString), 'getListIndicesSymmetricInGroupOfFolds'))
del ImaString

ImaString: str = f"""
def {defaultA007822['function']['filterAsymmetricFolds']}({defaultA007822['variable']['stateInstance']}: {defaultA007822['variable']['stateDataclass']}) -> {defaultA007822['variable']['stateDataclass']}:
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
