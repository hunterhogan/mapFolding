"""Generate reusable Codon JIT algorithm modules."""

from __future__ import annotations

from astToolkit import Be, DOT, Grab, Make, NodeChanger, NodeTourist, Then
from astToolkit.containers import astModuleToIngredientsFunction, IngredientsFunction, IngredientsModule, LedgerOfImports
from astToolkit.transformationTools import inlineFunctionDef
from copy import deepcopy
from functools import partial
from hunterMakesPy import raiseIfNone
from mapFolding import packageSettings
from mapFolding.someAssemblyRequired import default, Default, defaultCodon, IfThis
from mapFolding.someAssemblyRequired.codon.toolkitCodon import castParameterScalarForCodon, decorateCallableWithCodon, identifierModuleNumpy
from mapFolding.someAssemblyRequired.toolkitMakeModules import findDataclass, getLogicalPath, getModule, getPathFilename
from mapFolding.someAssemblyRequired.transformationTools import (
	removeDataclassFromFunction, shatter_dataclassesDOTdataclass, unpackDataclassCallFunctionRepackDataclass)
from typing import cast, TYPE_CHECKING

if TYPE_CHECKING:
	from astToolkit import identifierDotAttribute
	from pathlib import PurePath
	import ast

def codonJitOnFunction(astModule: ast.Module
						, identifierModule: str
						, identifierCallable: str | None = None
						, logicalPathInfix: identifierDotAttribute | None = None
						, sourceCallableDispatcher: str | None = None
						, identifiers: Default = default
						) -> PurePath:
	"""Generate a Codon JIT module from a dataclass-based counting function."""
	sourceCallableIdentifier: str = identifiers['function']['counting']
	targetCallableRoot: str = identifierCallable or f'{sourceCallableIdentifier}{identifierModule[0].upper()}{identifierModule[1:]}Codon'
	targetModuleLogicalPath: identifierDotAttribute = getLogicalPath(
		packageSettings.identifierPackage, logicalPathInfix, identifierModule)
	targetCallableIdentifier: str = f'{targetCallableRoot}__{targetModuleLogicalPath.replace('.', '_')}'
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))
	Grab.nameAttribute(Then.replaceWith(targetCallableIdentifier))(ingredientsFunction.astFunctionDef)

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
		NodeChanger(
			Be.Call.funcIs(Be.Name.idIs(IfThis.isIdentifier(sourceCallableIdentifier)))
			, doThat=Grab.funcAttribute(Grab.idAttribute(Then.replaceWith(targetCallableIdentifier)))
		).visit(ingredientsFunctionDispatcher.astFunctionDef)
		assignmentDispatcher: ast.Assign | None = NodeTourist(
			Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier))
			, doThat=Then.extractIt
		).captureLastMatch(ingredientsFunctionDispatcher.astFunctionDef)
		message: str = (f"I could not find an `ast.Assign` whose value calls `{targetCallableIdentifier = }` in "
			f"`{sourceCallableDispatcher = }`, but I need a direct assignment to transform the dispatcher.")
		raiseIfNone(assignmentDispatcher, message)
		ingredientsFunctionDispatcher = unpackDataclassCallFunctionRepackDataclass(
			ingredientsFunctionDispatcher, targetCallableIdentifier, shatteredDataclass)

		tupleReturned: ast.expr | None = NodeTourist(
			Be.Return.valueIs(Be.Tuple)
			, doThat=Then.extractIt(DOT.value)
		).captureLastMatch(ingredientsFunction.astFunctionDef)
		message = (f"I could not find an `ast.Return` with an `ast.Tuple` value in `{targetCallableIdentifier = }`, "
			"but I need its fields to construct the dispatcher boundary.")
		astTupleReturned: ast.Tuple = cast('ast.Tuple', raiseIfNone(tupleReturned, message))
		astTupleAssigned: ast.Tuple = deepcopy(astTupleReturned)
		Grab.ctxAttribute(Then.replaceWith(Make.Store()))(astTupleAssigned)
		NodeChanger(
			Be.Name
			, doThat=Grab.ctxAttribute(Then.replaceWith(Make.Store()))
		).visit(astTupleAssigned)
		Grab.ctxAttribute(Then.replaceWith(Make.Load()))(astTupleReturned)
		NodeChanger(
			Be.Name
			, doThat=Grab.ctxAttribute(Then.replaceWith(Make.Load()))
		).visit(astTupleReturned)
		assignmentShattered: ast.Assign | None = NodeTourist(
			Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier))
			, doThat=Then.extractIt
		).captureLastMatch(ingredientsFunctionDispatcher.astFunctionDef)
		message = (f"I could not find the shattered `ast.Assign` that calls `{targetCallableIdentifier = }` in "
			f"`{sourceCallableDispatcher = }`, but I need it to add the Codon scalar boundary.")
		raiseIfNone(assignmentShattered, message)
		NodeChanger(
			Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier))
			, doThat=Then.replaceWith(Make.Assign(
				[astTupleAssigned]
				, value=Make.Call(
					Make.Name(targetCallableIdentifier)
					, list(map(
						partial(castParameterScalarForCodon, shatteredDataclass)
						, cast('list[ast.Name]', astTupleReturned.elts)
					))
				)
			))
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
