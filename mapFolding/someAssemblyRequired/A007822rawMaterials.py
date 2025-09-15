from astToolkit import extractFunctionDef, IngredientsFunction, IngredientsModule, LedgerOfImports, Make  # noqa: D100
from hunterMakesPy import raiseIfNone
from mapFolding.someAssemblyRequired.infoBooth import dataclassInstanceIdentifierDEFAULT, theCountingIdentifierDEFAULT
from pprint import pprint
import ast

identifierDataclass: str = dataclassInstanceIdentifierDEFAULT
identifierCounting: str = theCountingIdentifierDEFAULT

identifier_filterAsymmetricFolds = 'filterAsymmetricFolds'
identifier_getAsymmetricFoldsTotal = 'getAsymmetricFoldsTotal'
identifier_initializeConcurrencyManager = 'initializeConcurrencyManager'
identifier_processCompletedFutures = '_processCompletedFutures'

ImaString = f"""
def {identifier_filterAsymmetricFolds}({identifierDataclass}: MapFoldingState) -> MapFoldingState:
	{identifierDataclass}.indexLeaf = 0
	leafConnectee = 0
	while leafConnectee < {identifierDataclass}.leavesTotal + 1:
		leafNumber = int({identifierDataclass}.leafBelow[{identifierDataclass}.indexLeaf])
		{identifierDataclass}.leafComparison[leafConnectee] = (leafNumber - {identifierDataclass}.indexLeaf + {identifierDataclass}.leavesTotal) % {identifierDataclass}.leavesTotal
		{identifierDataclass}.indexLeaf = leafNumber
		leafConnectee += 1

	indexInMiddle = {identifierDataclass}.leavesTotal // 2
	{identifierDataclass}.indexMiniGap = 0
	while {identifierDataclass}.indexMiniGap < {identifierDataclass}.leavesTotal + 1:
		ImaSymmetricFold = True
		leafConnectee = 0
		while leafConnectee < indexInMiddle:
			if {identifierDataclass}.leafComparison[({identifierDataclass}.indexMiniGap + leafConnectee) % ({identifierDataclass}.leavesTotal + 1)] != {identifierDataclass}.leafComparison[({identifierDataclass}.indexMiniGap + {identifierDataclass}.leavesTotal - 1 - leafConnectee) % ({identifierDataclass}.leavesTotal + 1)]:
				ImaSymmetricFold = False
				break
			leafConnectee += 1
		if ImaSymmetricFold:
			{identifierDataclass}.{identifierCounting} += 1
		{identifierDataclass}.indexMiniGap += 1

	return {identifierDataclass}
"""

FunctionDef_filterAsymmetricFolds: ast.FunctionDef = raiseIfNone(extractFunctionDef(ast.parse(ImaString), identifier_filterAsymmetricFolds))
del ImaString

ImaString = f"{identifierDataclass} = {identifier_filterAsymmetricFolds}({identifierDataclass})"
A007822incrementCount = ast.parse(ImaString).body[0]
del ImaString

ImaString = f'{identifierDataclass}.{identifierCounting} = ({identifierDataclass}.{identifierCounting} + 1) // 2'
A007822adjustFoldsTotal = ast.parse(ImaString).body[0]
del ImaString

astExprCall_filterAsymmetricFoldsDataclass: ast.Expr = Make.Expr(Make.Call(Make.Name(identifier_filterAsymmetricFolds), listParameters=[Make.Attribute(Make.Name(identifierDataclass), 'leafBelow')]))
# ----------------- Asynchronous --------------------------------------------------------------------------------------
astExprCall_filterAsymmetricFoldsLeafBelow: ast.Expr = Make.Expr(Make.Call(Make.Name(identifier_filterAsymmetricFolds), listParameters=[Make.Name('leafBelow')]))

AssignTotal2CountingIdentifier: ast.Assign = Make.Assign(
	[Make.Attribute(Make.Name(identifierDataclass), identifierCounting, context=Make.Store())]
	, value=Make.Call(Make.Name(identifier_getAsymmetricFoldsTotal))
)

astExprCall_initializeConcurrencyManager = Make.Expr(Make.Call(Make.Name(identifier_initializeConcurrencyManager)))

