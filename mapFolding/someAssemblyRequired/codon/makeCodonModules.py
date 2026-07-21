"""Generate reusable Codon JIT algorithm modules."""

from __future__ import annotations

from astToolkit import Be, DOT, Grab, Make, NodeChanger, NodeTourist, Then
from astToolkit.containers import astModuleToIngredientsFunction, IngredientsFunction, IngredientsModule
from copy import deepcopy
from functools import partial
from hunterMakesPy import raiseIfNone
from mapFolding import packageSettings
from mapFolding.someAssemblyRequired import default, Default, defaultCodon, IfThis
from mapFolding.someAssemblyRequired.codon.toolkitCodon import castParameterScalarForCodon, decorateCallableWithCodon, identifierModuleNumpy
from mapFolding.someAssemblyRequired.toolkitMakeModules import findDataclass, getModule, getPathFilename
from mapFolding.someAssemblyRequired.transformationTools import (
	removeDataclassFromFunction, shatter_dataclassesDOTdataclass, unpackDataclassCallFunctionRepackDataclass)
from typing import cast, TYPE_CHECKING

if TYPE_CHECKING:
	from astToolkit import identifierDotAttribute
	from pathlib import PurePath
	import ast

def codonJitOnFunction(
	astModule: ast.Module
	, identifierModule: str
	, identifierCallable: str | None = None
	, logicalPathInfix: identifierDotAttribute | None = None
	, sourceCallableDispatcher: str | None = None
	, identifiers: Default = default
) -> PurePath:
	"""Generate a Codon JIT module from a dataclass-based counting function."""
	ingredientsFunction: IngredientsFunction = astModuleToIngredientsFunction(astModule, identifiers['function']['counting'])

	shatteredDataclass = shatter_dataclassesDOTdataclass(*findDataclass(ingredientsFunction))
	shatteredDataclass.imports.removeImportFromModule('numpy')
	ingredientsFunction.imports.update(shatteredDataclass.imports)
	ingredientsFunction = removeDataclassFromFunction(ingredientsFunction, shatteredDataclass)
	ingredientsFunction.removeUnusedParameters()
	ingredientsFunction = decorateCallableWithCodon(ingredientsFunction)

	ingredientsModule = IngredientsModule(ingredientsFunction)

	if sourceCallableDispatcher is not None:
		ingredientsFunctionDispatcher: IngredientsFunction = astModuleToIngredientsFunction(astModule, sourceCallableDispatcher)
		ingredientsFunctionDispatcher.imports.update(shatteredDataclass.imports)
		ingredientsFunctionDispatcher.imports.addImport_asStr(identifierModuleNumpy)
		targetCallableIdentifier = ingredientsFunction.astFunctionDef.name
		NodeChanger(
			Be.Call.funcIs(Be.Name.idIs(IfThis.isIdentifier(identifiers['function']['counting'])))
			, doThat=Grab.funcAttribute(Grab.idAttribute(Then.replaceWith(targetCallableIdentifier)))
		).visit(ingredientsFunctionDispatcher.astFunctionDef)
		assignmentDispatcher: ast.Assign = raiseIfNone(
			NodeTourist(Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier)), doThat=Then.extractIt).captureLastMatch(
				ingredientsFunctionDispatcher.astFunctionDef
			)
		)

		ingredientsFunctionDispatcher = unpackDataclassCallFunctionRepackDataclass(
			ingredientsFunctionDispatcher, targetCallableIdentifier, shatteredDataclass
		)

		tupleReturned: ast.expr = raiseIfNone(
			NodeTourist(Be.Return.valueIs(Be.Tuple), doThat=Then.extractIt(DOT.value)).captureLastMatch(ingredientsFunction.astFunctionDef)
		)
		astTupleReturned: ast.Tuple = cast('ast.Tuple', tupleReturned)
		astTupleAssigned: ast.Tuple = deepcopy(astTupleReturned)
		Grab.ctxAttribute(Then.replaceWith(Make.Store()))(astTupleAssigned)
		NodeChanger(Be.Name, doThat=Grab.ctxAttribute(Then.replaceWith(Make.Store()))).visit(astTupleAssigned)
		Grab.ctxAttribute(Then.replaceWith(Make.Load()))(astTupleReturned)
		NodeChanger(Be.Name, doThat=Grab.ctxAttribute(Then.replaceWith(Make.Load()))).visit(astTupleReturned)
		assignmentShattered: ast.Assign = raiseIfNone(
			NodeTourist(Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier)), doThat=Then.extractIt).captureLastMatch(
				ingredientsFunctionDispatcher.astFunctionDef
			)
		)

		NodeChanger(
			Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier))
			, doThat=Then.replaceWith(
				Make.Assign(
					[astTupleAssigned]
					, value=Make.Call(
						Make.Name(targetCallableIdentifier)
						, list(map(partial(castParameterScalarForCodon, shatteredDataclass), cast('list[ast.Name]', astTupleReturned.elts)))
					)
				)
			)
		).visit(ingredientsFunctionDispatcher.astFunctionDef)

		ingredientsModule.appendIngredientsFunction(ingredientsFunctionDispatcher)

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, identifierModule)
	ingredientsModule.write_astModule(pathFilename, identifierPackage=packageSettings.identifierPackage)
	return pathFilename

def makeTheorem2Codon() -> PurePath:
	"""Generate the Codon JIT version of the trimmed Theorem 2 algorithm."""
	return codonJitOnFunction(
		getModule(logicalPathInfix=default['logicalPath']['synthetic'], identifierModule=defaultCodon['module']['source'])
		, defaultCodon['module']['algorithm']
		, logicalPathInfix=defaultCodon['logicalPath']['synthetic']
		, sourceCallableDispatcher=defaultCodon['function']['dispatcher']
		, identifiers=defaultCodon
	)

if __name__ == '__main__':
	makeTheorem2Codon()
