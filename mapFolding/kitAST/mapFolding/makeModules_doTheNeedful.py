"""Make functions that are complementary to the `count` function and are often called by `doTheNeedful`."""
from __future__ import annotations

from astToolkit import Grab, identifierDotAttribute, NodeChanger, Then
from astToolkit.containers import IngredientsFunction, IngredientsModule, LedgerOfImports
from astToolkit.transformationTools import inlineFunctionDef
from mapFolding.kitAST import IfThis
from mapFolding.kitAST.kitMakeModules import findDataclass, getPathFilename
from mapFolding.kitAST.theSSOT import default
from typing import TYPE_CHECKING
import ast

if TYPE_CHECKING:
	from mapFolding.theTypes import Default
	from os import PathLike
	from pathlib import PurePath
	from typing import Any

def makeInitializeState(astModule: ast.Module, identifiers: Default | None = None, **keywordArguments: Any) -> PurePath:
	"""Generate initialization module for counting variable setup.

	(AI generated docstring)

	Creates a specialized module containing initialization logic for the counting variables used in
	map folding computations. The generated function transforms the original algorithm's loop
	conditions to use equality comparisons instead of greater-than comparisons, optimizing the
	initialization phase.

	This transformation is particularly important for ensuring that counting variables are properly
	initialized before the main computational loops begin executing.

	Parameters
	----------
	astModule : ast.Module
		Source module containing the base algorithm.
	identifierModule : str
		Name for the generated initialization module.
	identifierCallable : str | None = None
		Name for the initialization function.
	logicalPathInfix : identifierDotAttribute | None = None
		Directory path for organizing the generated module.
	sourceCallableDispatcher : str | None = None
		Optional dispatcher function identifier.

	Returns
	-------
	pathFilename : PurePath
		Filesystem path where the initialization module was written.

	"""
	identifiers = identifiers or default
	identifierCallableSource: identifierDotAttribute = keywordArguments.get('identifierCallableSource') or identifiers['function']['counting']
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(identifierCallableSource, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = keywordArguments.get('identifierCallable') or identifiers['function'].get('initializeState') or identifierCallableSource

	_logicalPathDataclass, _identifierDataclass, identifierDataclassInstance = findDataclass(ingredientsFunction)
	identifierCounting: identifierDotAttribute = keywordArguments.get('identifierCounting') or identifiers['variable']['counting']

	NodeChanger(findThis=IfThis.isWhileAttributeNamespaceIdentifierGreaterThan0(identifierDataclassInstance, 'leaf1ndex')
		, doThat=Grab.testAttribute(Grab.andDoAllOf([
			Grab.opsAttribute(Then.replaceWith([ast.Eq()]))
			, Grab.leftAttribute(Grab.attrAttribute(Then.replaceWith(identifierCounting)))]))
	).visit(ingredientsFunction.astFunctionDef.body[0])

	pathRoot: PathLike[str] = keywordArguments.get('pathRoot') or identifiers['filesystem']['pathRoot']
	logicalPathInfix: identifierDotAttribute = keywordArguments.get('logicalPathInfix') or identifiers['logicalPath']['synthetic']
	identifierModule: str = keywordArguments.get('identifierModule') or identifiers['module']['initializeState']

	pathFilename: PurePath = getPathFilename(pathRoot, logicalPathInfix, identifierModule)
	identifierPackage: str = keywordArguments.get('identifierPackage') or identifiers['module']['identifierPackage']
	IngredientsModule(ingredientsFunction).write_astModule(pathFilename, identifierPackage)

	return pathFilename
