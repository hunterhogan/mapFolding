from astToolkit import extractFunctionDef  # noqa: D100
from hunterMakesPy import raiseIfNone
from mapFolding.someAssemblyRequired.infoBooth import (
	dataclassInstanceIdentifierDEFAULT, sourceCallableIdentifierDEFAULT, theCountingIdentifierDEFAULT)
import ast

Z0Z_identifier = 'filterAsymmetricFolds'
ImaString = f"""
def {Z0Z_identifier}(state: MapFoldingState) -> MapFoldingState:
	state.indexLeaf = 0
	leafConnectee = 0
	while leafConnectee < state.leavesTotal + 1:
		leafNumber = int(state.leafBelow[state.indexLeaf])
		state.leafComparison[leafConnectee] = (leafNumber - state.indexLeaf + state.leavesTotal) % state.leavesTotal
		state.indexLeaf = leafNumber
		leafConnectee += 1

	indexInMiddle = state.leavesTotal // 2
	state.indexMiniGap = 0
	while state.indexMiniGap < state.leavesTotal + 1:
		ImaSymmetricFold = True
		leafConnectee = 0
		while leafConnectee < indexInMiddle:
			if state.leafComparison[(state.indexMiniGap + leafConnectee) % (state.leavesTotal + 1)] != state.leafComparison[(state.indexMiniGap + state.leavesTotal - 1 - leafConnectee) % (state.leavesTotal + 1)]:
				ImaSymmetricFold = False
				break
			leafConnectee += 1
		if ImaSymmetricFold:
			state.groupsOfFolds += 1
		state.indexMiniGap += 1

	return state
"""

FunctionDef_filterAsymmetricFolds: ast.FunctionDef = raiseIfNone(extractFunctionDef(ast.parse(ImaString), Z0Z_identifier))
del ImaString

ImaString = f"{dataclassInstanceIdentifierDEFAULT} = {Z0Z_identifier}({dataclassInstanceIdentifierDEFAULT})"
Z0Z_incrementCount = ast.parse(ImaString).body[0]
del ImaString


ImaString = 'state.groupsOfFolds = (state.groupsOfFolds + 1) // 2'
Z0Z_adjustFoldsTotal = ast.parse(ImaString).body[0]
del ImaString

