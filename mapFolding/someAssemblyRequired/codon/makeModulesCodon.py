"""Generate Codon JIT algorithm modules."""
from __future__ import annotations

from astToolkit import Be, DOT, Grab, identifierDotAttribute, Make, NodeChanger, Then
from astToolkit.containers import astModuleToIngredientsFunction, IngredientsFunction, IngredientsModule, LedgerOfImports
from collections.abc import Callable  # ruff:ignore[typing-only-standard-library-import]
from copy import deepcopy
from mapFolding.someAssemblyRequired import Default, default, IfThis, ShatteredDataclass
from mapFolding.someAssemblyRequired.codon.kitCodon import decorateCallableWithCodon, variableCompatibility
from mapFolding.someAssemblyRequired.kitMakeModules import findDataclass, getModule, getPathFilename
from mapFolding.someAssemblyRequired.kitTransformations import (
	removeDataclassFromFunction, shatter_dataclassesDOTdataclass, unpackDataclassCallFunctionRepackDataclass)
from mapFolding.theSSOT import settingsPackage
from more_itertools import consume
from pathlib import Path, PurePath
import ast  # ruff:ignore[typing-only-standard-library-import]

def _dataclassImports(
	logicalPathDataclass: identifierDotAttribute
	, shatteredDataclass: ShatteredDataclass
) -> LedgerOfImports:
	importsDataclass: LedgerOfImports = deepcopy(shatteredDataclass.imports)

	def isNotDataclassModule(logicalPathModule: identifierDotAttribute) -> bool:
		return logicalPathModule != logicalPathDataclass

	consume(map(
		importsDataclass.removeImportFromModule
		, filter(
			isNotDataclassModule
			, importsDataclass.exportListModuleIdentifiers()
		)
	))
	return importsDataclass

def _parameterAssignmentTarget(parameter: ast.arg) -> ast.Name:
	return Make.Name(parameter.arg, Make.Store())

def _parameterFromDataclass(
	parameter: ast.arg
	, shatteredDataclass: ShatteredDataclass
) -> ast.expr:
	parameterName: ast.Name = Make.Name(parameter.arg)
	fieldAssignment: ast.AnnAssign | ast.Assign = shatteredDataclass.field2AnnAssign[parameter.arg]
	if not Be.AnnAssign(fieldAssignment):
		return parameterName
	fieldValue: ast.expr | None = fieldAssignment.value
	if fieldValue is None or not Be.Call(fieldValue):
		return parameterName
	if not IfThis.unparseIs(fieldAssignment.annotation)(fieldValue.func):
		return parameterName
	return Make.Call(deepcopy(fieldAssignment.annotation), [parameterName])

def _dispatcherAssignment(
	identifierCallable: str
	, parameters: list[ast.arg]
	, shatteredDataclass: ShatteredDataclass
) -> ast.Assign:
	def parameterFromDataclass(parameter: ast.arg) -> ast.expr:
		return _parameterFromDataclass(parameter, shatteredDataclass)

	assignmentTarget: ast.Tuple = Make.Tuple(
		list(map(_parameterAssignmentTarget, parameters))
		, Make.Store()
	)
	return Make.Assign(
		[assignmentTarget]
		, Make.Call(
			Make.Name(identifierCallable)
			, list(map(parameterFromDataclass, parameters))
		)
	)

def _makeDispatcherAssignment(
	identifierCallable: str
	, parameters: list[ast.arg]
	, shatteredDataclass: ShatteredDataclass
) -> Callable[[ast.Assign], ast.Assign]:
	def makeDispatcherAssignment(_assignment: ast.Assign) -> ast.Assign:
		return _dispatcherAssignment(
			identifierCallable, parameters, shatteredDataclass)
	return makeDispatcherAssignment

def _restoreFieldsExcludedFromInit(
	identifierDataclassInstance: str
	, shatteredDataclass: ShatteredDataclass
) -> list[ast.Assign]:
	identifiersInitialized: set[str | None] = set(map(
		DOT.arg, shatteredDataclass.list_keyword_field__field4init))

	def isExcludedFromInit(identifier: str) -> bool:
		return identifier not in identifiersInitialized

	def restoreField(identifier: str) -> ast.Assign:
		return Make.Assign(
			[Make.Attribute(
				Make.Name(identifierDataclassInstance)
				, identifier
				, context=Make.Store()
			)]
			, Make.Name(identifier)
		)

	return list(map(
		restoreField
		, filter(isExcludedFromInit, shatteredDataclass.field2AnnAssign)
	))

def codonJitOnFunction(
	astModule: ast.Module
	, identifierModule: str
	, identifierCallable: str | None = None
	, logicalPathInfix: identifierDotAttribute | None = None
	, sourceCallableDispatcher: str | None = None
	, identifiers: Default | None = None
) -> PurePath:
	"""Generate a Codon JIT module from a dataclass-based function."""
	identifiersCurrent: Default = identifiers or default
	sourceCallableIdentifier: str = identifiersCurrent['function']['counting']
	ingredientsFunction: IngredientsFunction = astModuleToIngredientsFunction(astModule, sourceCallableIdentifier)

	logicalPathDataclass, identifierDataclass, identifierDataclassInstance = findDataclass(ingredientsFunction)
	targetCallableIdentifier: str = identifierCallable or f'{sourceCallableIdentifier}{identifierDataclass}'
	Grab.nameAttribute(Then.replaceWith(targetCallableIdentifier))(ingredientsFunction.astFunctionDef)
	shatteredDataclass = shatter_dataclassesDOTdataclass(
		logicalPathDataclass, identifierDataclass, identifierDataclassInstance)
	ingredientsFunction = removeDataclassFromFunction(ingredientsFunction, shatteredDataclass)
	ingredientsFunction.removeUnusedParameters()
	ingredientsFunction = variableCompatibility(
		ingredientsFunction, ingredientsFunction.astFunctionDef.args.args)
	ingredientsFunction = decorateCallableWithCodon(ingredientsFunction)
	ingredientsModule = IngredientsModule(ingredientsFunction)

	if sourceCallableDispatcher is not None:
		ingredientsFunctionDispatcher: IngredientsFunction = astModuleToIngredientsFunction(astModule, sourceCallableDispatcher)
		ingredientsFunctionDispatcher.imports.update(_dataclassImports(
			logicalPathDataclass, shatteredDataclass))
		NodeChanger(
			Be.Call.funcIs(Be.Name.idIs(IfThis.isIdentifier(sourceCallableIdentifier)))
			, Grab.funcAttribute(Grab.idAttribute(Then.replaceWith(targetCallableIdentifier)))
		).visit(ingredientsFunctionDispatcher.astFunctionDef)
		ingredientsFunctionDispatcher = unpackDataclassCallFunctionRepackDataclass(
			ingredientsFunctionDispatcher, targetCallableIdentifier, shatteredDataclass)
		NodeChanger(
			Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier))
			, _makeDispatcherAssignment(
				targetCallableIdentifier
				, ingredientsFunction.astFunctionDef.args.args
				, shatteredDataclass
			)
		).visit(ingredientsFunctionDispatcher.astFunctionDef)
		NodeChanger(
			Be.Assign.valueIs(IfThis.isCallIdentifier(identifierDataclass))
			, Then.insertThisBelow(_restoreFieldsExcludedFromInit(
				identifierDataclassInstance, shatteredDataclass))
		).visit(ingredientsFunctionDispatcher.astFunctionDef)
		ingredientsModule.appendIngredientsFunction(ingredientsFunctionDispatcher)

	pathFilename: PurePath = getPathFilename(
		settingsPackage.pathPackage, logicalPathInfix, identifierModule)
	Path(pathFilename).parent.mkdir(parents=True, exist_ok=True)
	ingredientsModule.write_astModule(
		pathFilename, identifierPackage=settingsPackage.identifierPackage)
	return pathFilename

def makeTheorem2Codon(identifiers: Default | None = None) -> PurePath:
	"""Generate the Codon JIT version of a configured trimmed Theorem 2 algorithm."""
	identifiersCurrent: Default = identifiers or default
	logicalPathSynthetic: identifierDotAttribute = identifiersCurrent['logicalPath']['synthetic']
	return codonJitOnFunction(
		getModule(logicalPathInfix=logicalPathSynthetic, identifierModule='theorem2Trimmed')
		, 'theorem2'
		, logicalPathInfix=f'{logicalPathSynthetic}.codon'
		, sourceCallableDispatcher=identifiersCurrent['function']['dispatcher']
		, identifiers=identifiersCurrent
	)

if __name__ == '__main__':
	makeTheorem2Codon()
