"""Generate Codon JIT algorithm modules."""
from __future__ import annotations

from astToolkit import Be, DOT, Grab, Make, NodeChanger, NodeTourist, Then
from astToolkit.containers import astModuleToIngredientsFunction, IngredientsFunction, IngredientsModule
from copy import deepcopy
from functools import partial
from hunterMakesPy import raiseIfNone
from mapFolding.someAssemblyRequired import default, IfThis
from mapFolding.someAssemblyRequired.codon.kitCodon import (
	decorateCallableWithCodon, getIntegerArrayDtypes, integerArraysCodonCompatible, parameterCodonCompatible)
from mapFolding.someAssemblyRequired.kitMakeModules import findDataclass, getLogicalPath, getModule, getPathFilename
from mapFolding.someAssemblyRequired.kitTransformations import (
	removeDataclassFromFunction, shatter_dataclassesDOTdataclass, unpackDataclassCallFunctionRepackDataclass)
from mapFolding.theSSOT import settingsPackage
from typing import TYPE_CHECKING

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
	, sourceCallableIdentifier: str = default['function']['counting']
) -> PurePath:
	"""Generate a Codon JIT module from a dataclass-based function."""
	ingredientsFunction: IngredientsFunction = astModuleToIngredientsFunction(astModule, sourceCallableIdentifier)
	targetCallableIdentifier: str = identifierCallable or sourceCallableIdentifier

	logicalPathDataclass, identifierDataclass, identifierDataclassInstance = findDataclass(ingredientsFunction)
	shatteredDataclass = shatter_dataclassesDOTdataclass(
		logicalPathDataclass, identifierDataclass, identifierDataclassInstance)
	shatteredDataclass.imports.removeImportFromModule('numpy')
	ingredientsFunction.imports.update(shatteredDataclass.imports)
	ingredientsFunction = removeDataclassFromFunction(ingredientsFunction, shatteredDataclass)
	ingredientsFunction.removeUnusedParameters()
	ingredientsFunction = integerArraysCodonCompatible(
		ingredientsFunction, getIntegerArrayDtypes(logicalPathDataclass, identifierDataclass))
	ingredientsFunction = decorateCallableWithCodon(ingredientsFunction)
	ingredientsModule = IngredientsModule(ingredientsFunction)

	if sourceCallableDispatcher is not None:
		ingredientsFunctionDispatcher: IngredientsFunction = astModuleToIngredientsFunction(astModule, sourceCallableDispatcher)
		ingredientsFunctionDispatcher.imports.update(shatteredDataclass.imports)
		NodeChanger(
			Be.Call.funcIs(Be.Name.idIs(IfThis.isIdentifier(sourceCallableIdentifier)))
			, doThat=Grab.funcAttribute(Grab.idAttribute(Then.replaceWith(targetCallableIdentifier)))
		).visit(ingredientsFunctionDispatcher.astFunctionDef)
		ingredientsFunctionDispatcher = unpackDataclassCallFunctionRepackDataclass(
			ingredientsFunctionDispatcher, targetCallableIdentifier, shatteredDataclass)

		astTupleReturned = raiseIfNone(NodeTourist(
			Be.Return.valueIs(Be.Tuple)
			, doThat=Then.extractIt(DOT.value)
		).captureLastMatch(ingredientsFunction.astFunctionDef))
		astTupleAssigned = deepcopy(astTupleReturned)

		NodeChanger(Be.Name, doThat=Grab.ctxAttribute(Then.replaceWith(Make.Store()))).visit(astTupleAssigned)

		NodeChanger(Be.Name, doThat=Grab.ctxAttribute(Then.replaceWith(Make.Load()))).visit(astTupleReturned)
		NodeChanger(
			Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier))
			, doThat=Then.replaceWith(Make.Assign(
				[astTupleAssigned]
				, value=Make.Call(
					Make.Name(targetCallableIdentifier)
					, list(map(partial(parameterCodonCompatible, shatteredDataclass), astTupleReturned.elts))
				)
			))
		).visit(ingredientsFunctionDispatcher.astFunctionDef)
		ingredientsModule.appendIngredientsFunction(ingredientsFunctionDispatcher)

	pathFilename: PurePath = getPathFilename(settingsPackage.pathPackage, logicalPathInfix, identifierModule)
	ingredientsModule.write_astModule(pathFilename, identifierPackage=settingsPackage.identifierPackage)
	return pathFilename

def makeTheorem2Codon() -> PurePath:
	"""Generate the Codon JIT version of the trimmed Theorem 2 algorithm."""
	pathFilename: PurePath = codonJitOnFunction(
		getModule(logicalPathInfix=default['logicalPath']['synthetic'], identifierModule='theorem2Trimmed')
		, 'theorem2'
		, logicalPathInfix=f'{default["logicalPath"]["synthetic"]}.codon'
		, sourceCallableDispatcher=default['function']['dispatcher']
	)
	return pathFilename

if __name__ == '__main__':
	makeTheorem2Codon()
