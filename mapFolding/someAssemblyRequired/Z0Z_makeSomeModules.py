from mapFolding import raiseIfNoneGitHubIssueNumber3, The
from mapFolding.someAssemblyRequired import (
	ast_Identifier,
	be,
	DOT,
	grab,
	ifThis,
	IngredientsFunction,
	IngredientsModule,
	LedgerOfImports,
	Make,
	NodeChanger,
	NodeTourist,
	parseLogicalPath2astModule,
	Then,
)
from mapFolding.someAssemblyRequired.transformationTools import inlineFunctionDef, write_astModule
from pathlib import PurePath
import ast

algorithmSourceModuleHARDCODED = 'daoOfMapFolding'
sourceCallableIdentifierHARDCODED = 'count'
logicalPathInfixHARDCODED: ast_Identifier = 'syntheticModules'
theCountingIdentifierHARDCODED: ast_Identifier = 'groupsOfFolds'

def makeInitializeGroupsOfFolds():
	callableIdentifierHARDCODED = 'initializeGroupsOfFolds'
	moduleIdentifierHARDCODED: ast_Identifier = 'initializeCount'

	algorithmSourceModule = algorithmSourceModuleHARDCODED
	sourceCallableIdentifier = sourceCallableIdentifierHARDCODED
	logicalPathSourceModule = '.'.join([The.packageName, algorithmSourceModule])

	callableIdentifier = callableIdentifierHARDCODED
	logicalPathInfix = logicalPathInfixHARDCODED
	moduleIdentifier = moduleIdentifierHARDCODED

	astModule = parseLogicalPath2astModule(logicalPathSourceModule)
	countInitializeIngredients = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule)
		, LedgerOfImports(astModule))

	countInitializeIngredients.astFunctionDef.name = callableIdentifier

	dataclassIdentifier = NodeTourist(be.arg, Then.extractIt(DOT.arg)).captureLastMatch(countInitializeIngredients.astFunctionDef)
	if dataclassIdentifier is None: raise raiseIfNoneGitHubIssueNumber3
	theCountingIdentifier = theCountingIdentifierHARDCODED

	findThis = ifThis.isWhileAttributeNamespace_IdentifierGreaterThan0(dataclassIdentifier, 'leaf1ndex')
	doThat = grab.testAttribute(grab.andDoAllOf([
		grab.opsAttribute(Then.replaceWith([ast.Eq()])), # type: ignore
		grab.leftAttribute(grab.attrAttribute(Then.replaceWith(theCountingIdentifier))) # type: ignore
	]))
	NodeChanger(findThis, doThat).visit(countInitializeIngredients.astFunctionDef.body[0])

	initializationModule = IngredientsModule(countInitializeIngredients)

	pathFilename = PurePath(The.pathPackage, logicalPathInfix, moduleIdentifier + The.fileExtension)

	write_astModule(initializationModule, pathFilename, The.packageName)

def makeTheorem2():
	moduleIdentifierHARDCODED: ast_Identifier = 'theorem2'

	algorithmSourceModule = algorithmSourceModuleHARDCODED
	sourceCallableIdentifier = sourceCallableIdentifierHARDCODED
	logicalPathSourceModule = '.'.join([The.packageName, algorithmSourceModule])

	logicalPathInfix = logicalPathInfixHARDCODED
	moduleIdentifier = moduleIdentifierHARDCODED

	astModule = parseLogicalPath2astModule(logicalPathSourceModule)
	countTheorem2 = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule)
		, LedgerOfImports(astModule))

	dataclassIdentifier = NodeTourist(be.arg, Then.extractIt(DOT.arg)).captureLastMatch(countTheorem2.astFunctionDef)
	if dataclassIdentifier is None: raise raiseIfNoneGitHubIssueNumber3

	findThis = ifThis.isWhileAttributeNamespace_IdentifierGreaterThan0(dataclassIdentifier, 'leaf1ndex')
	doThat = grab.testAttribute(grab.comparatorsAttribute(Then.replaceWith([Make.Constant(4)]))) # type: ignore
	NodeChanger(findThis, doThat).visit(countTheorem2.astFunctionDef)

	findThis = ifThis.isIfAttributeNamespace_IdentifierGreaterThan0(dataclassIdentifier, 'leaf1ndex')
	doThat = Then.extractIt(DOT.body)
	insertLeaf = NodeTourist(findThis, doThat).captureLastMatch(countTheorem2.astFunctionDef)
	findThis = ifThis.isIfAttributeNamespace_IdentifierGreaterThan0(dataclassIdentifier, 'leaf1ndex')
	doThat = Then.replaceWith(insertLeaf)
	NodeChanger(findThis, doThat).visit(countTheorem2.astFunctionDef)

	findThis = ifThis.isAttributeNamespace_IdentifierGreaterThan0(dataclassIdentifier, 'leaf1ndex')
	doThat = Then.removeIt
	NodeChanger(findThis, doThat).visit(countTheorem2.astFunctionDef)

	findThis = ifThis.isAttributeNamespace_IdentifierLessThanOrEqual(dataclassIdentifier, 'leaf1ndex')
	doThat = Then.removeIt
	NodeChanger(findThis, doThat).visit(countTheorem2.astFunctionDef)

	# state.groupsOfFolds *= 2
	theCountingIdentifier = theCountingIdentifierHARDCODED
	doubleTheCount = Make.AugAssign(Make.Attribute(ast.Name(dataclassIdentifier), theCountingIdentifier), ast.Mult(), Make.Constant(2))
	findThis = be.Return
	doThat = Then.insertThisAbove([doubleTheCount])
	NodeChanger(findThis, doThat).visit(countTheorem2.astFunctionDef)

	moduleTheorem2 = IngredientsModule(countTheorem2)

	pathFilename = PurePath(The.pathPackage, logicalPathInfix, moduleIdentifier + The.fileExtension)

	write_astModule(moduleTheorem2, pathFilename, The.packageName)

if __name__ == '__main__':
	makeInitializeGroupsOfFolds()
	makeTheorem2()
