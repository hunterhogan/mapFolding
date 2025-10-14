"""Make functions that are complementary to the `count` function and are often called by `doTheNeedful`."""
from astToolkit import Be, DOT, Grab, identifierDotAttribute, NodeChanger, NodeTourist, Then
from astToolkit.containers import IngredientsFunction, IngredientsModule, LedgerOfImports
from astToolkit.transformationTools import inlineFunctionDef
from hunterMakesPy import raiseIfNone
from mapFolding import packageSettings
from mapFolding.someAssemblyRequired import default, Default, IfThis
from mapFolding.someAssemblyRequired.toolkitMakeModules import getPathFilename
from pathlib import PurePath
import ast

def makeInitializeState(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: identifierDotAttribute | None = None, sourceCallableDispatcher: str | None = None, identifiers: Default | None = None) -> PurePath:  # noqa: ARG001
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
	dictionaryIdentifiers: Default = identifiers or default
	sourceCallableIdentifier: identifierDotAttribute = dictionaryIdentifiers['function']['counting']
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = callableIdentifier or sourceCallableIdentifier

	dataclassInstanceIdentifier: identifierDotAttribute = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))
	theCountingIdentifier: identifierDotAttribute = dictionaryIdentifiers['variable']['counting']

	findThis = IfThis.isWhileAttributeNamespaceIdentifierGreaterThan0(dataclassInstanceIdentifier, 'leaf1ndex')
	doThat = Grab.testAttribute(Grab.andDoAllOf([Grab.opsAttribute(Then.replaceWith([ast.Eq()])), Grab.leftAttribute(Grab.attrAttribute(Then.replaceWith(theCountingIdentifier)))]))
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef.body[0])

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)
	IngredientsModule(ingredientsFunction).write_astModule(pathFilename, identifierPackage=packageSettings.identifierPackage)

	return pathFilename
