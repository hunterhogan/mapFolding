"""Make functions that are complementary to the `count` function and are often called by `doTheNeedful`."""
from astToolkit import (
	astModuleToIngredientsFunction, Be, DOT, extractFunctionDef, Grab, identifierDotAttribute, IngredientsFunction,
	IngredientsModule, LedgerOfImports, Make, NodeChanger, NodeTourist, parseLogicalPath2astModule, Then)
from astToolkit.transformationTools import inlineFunctionDef, write_astModule
from hunterMakesPy import raiseIfNone
from mapFolding import packageSettings
from mapFolding.someAssemblyRequired import (
	identifierCallableSourceDEFAULT, identifierCallableSourceDispatcherDEFAULT, identifierCountingDEFAULT,
	identifierModuleDataPackingDEFAULT, identifierModuleSourceAlgorithmDEFAULT, IfThis, logicalPathInfixAlgorithmDEFAULT,
	logicalPathInfixDEFAULT, ShatteredDataclass)
from mapFolding.someAssemblyRequired.toolkitMakeModules import findDataclass, getPathFilename
from mapFolding.someAssemblyRequired.transformationTools import (
	shatter_dataclassesDOTdataclass, unpackDataclassCallFunctionRepackDataclass)
from os import PathLike
from pathlib import PurePath
from typing import cast
import ast

def makeInitializeState(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:  # noqa: ARG001
	"""Generate initialization module for counting variable setup.

	(AI generated docstring)

	Creates a specialized module containing initialization logic for the counting variables
	used in map folding computations. The generated function transforms the original
	algorithm's loop conditions to use equality comparisons instead of greater-than
	comparisons, optimizing the initialization phase.

	This transformation is particularly important for ensuring that counting variables
	are properly initialized before the main computational loops begin executing.

	Parameters
	----------
	astModule : ast.Module
		Source module containing the base algorithm.
	moduleIdentifier : str
		Name for the generated initialization module.
	callableIdentifier : str | None = None
		Name for the initialization function.
	logicalPathInfix : PathLike[str] | PurePath | str | None = None
		Directory path for organizing the generated module.
	sourceCallableDispatcher : str | None = None
		Optional dispatcher function identifier.

	Returns
	-------
	pathFilename : PurePath
		Filesystem path where the initialization module was written.

	"""
	sourceCallableIdentifier: identifierDotAttribute = identifierCallableSourceDEFAULT
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = callableIdentifier or sourceCallableIdentifier

	dataclassInstanceIdentifier: identifierDotAttribute = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))
	theCountingIdentifier: identifierDotAttribute = identifierCountingDEFAULT

	findThis = IfThis.isWhileAttributeNamespaceIdentifierGreaterThan0(dataclassInstanceIdentifier, 'leaf1ndex')
	doThat = Grab.testAttribute(Grab.andDoAllOf([Grab.opsAttribute(Then.replaceWith([ast.Eq()])), Grab.leftAttribute(Grab.attrAttribute(Then.replaceWith(theCountingIdentifier)))]))
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef.body[0])

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)
	write_astModule(IngredientsModule(ingredientsFunction), pathFilename, packageSettings.identifierPackage)

	return pathFilename


def makeUnRePackDataclass(astImportFrom: ast.ImportFrom, moduleIdentifier: str = identifierModuleDataPackingDEFAULT) -> None:
	"""Generate interface module for dataclass unpacking and repacking operations.

	Parameters
	----------
	astImportFrom : ast.ImportFrom
		Import statement specifying the target optimized function to call.

	Returns
	-------
	None
		The generated module is written directly to the filesystem.

	"""
	callableIdentifierHARDCODED: str = 'sequential'

	algorithmSourceModule: identifierDotAttribute = identifierModuleSourceAlgorithmDEFAULT
	callableIdentifier: identifierDotAttribute = callableIdentifierHARDCODED
	logicalPathInfix: identifierDotAttribute = logicalPathInfixDEFAULT
	logicalPathInfixAlgorithm: identifierDotAttribute = logicalPathInfixAlgorithmDEFAULT
	sourceCallableIdentifier: identifierDotAttribute = identifierCallableSourceDispatcherDEFAULT

	logicalPathSourceModule: identifierDotAttribute = '.'.join([packageSettings.identifierPackage, logicalPathInfixAlgorithm, algorithmSourceModule])  # noqa: FLY002

	ingredientsFunction: IngredientsFunction = astModuleToIngredientsFunction(parseLogicalPath2astModule(logicalPathSourceModule), sourceCallableIdentifier)
	ingredientsFunction.astFunctionDef.name = callableIdentifier

	shatteredDataclass: ShatteredDataclass = shatter_dataclassesDOTdataclass(*findDataclass(ingredientsFunction))

	ingredientsFunction.imports.update(shatteredDataclass.imports)
	ingredientsFunction.imports.addAst(astImportFrom)
	targetCallableIdentifier: str = astImportFrom.names[0].name
	ingredientsFunction = raiseIfNone(unpackDataclassCallFunctionRepackDataclass(ingredientsFunction, targetCallableIdentifier, shatteredDataclass))
	targetFunctionDef: ast.FunctionDef = raiseIfNone(extractFunctionDef(parseLogicalPath2astModule(raiseIfNone(astImportFrom.module)), targetCallableIdentifier))
	astTuple: ast.Tuple = cast('ast.Tuple', raiseIfNone(NodeTourist(Be.Return.valueIs(Be.Tuple)
			, doThat=Then.extractIt(DOT.value)).captureLastMatch(targetFunctionDef)))
	astTuple.ctx = Make.Store()

	changeAssignCallToTarget = NodeChanger(
		findThis = Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier))
		, doThat = Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(targetCallableIdentifier), astTuple.elts))))
	changeAssignCallToTarget.visit(ingredientsFunction.astFunctionDef)

	ingredientsModule = IngredientsModule(ingredientsFunction)
	ingredientsModule.removeImportFromModule('numpy')

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

	write_astModule(ingredientsModule, pathFilename, packageSettings.identifierPackage)



