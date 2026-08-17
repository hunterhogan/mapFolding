"""Make the `count` function for an algorithm.

These transformation functions will work on at least two different algorithms. If a transformation
function only works on a specific type of algorithm, it will be in a subdirectory.
"""
from __future__ import annotations

from astToolkit import (
	Be, DOT, extractClassDef, Grab, hasDOTbody, identifierDotAttribute, Make, NodeChanger, NodeTourist, parseLogicalPath2astModule, Then)
from astToolkit.containers import astModuleToIngredientsFunction, IngredientsFunction, IngredientsModule, LedgerOfImports
from astToolkit.transformationTools import inlineFunctionDef
from hunterMakesPy import raiseIfNone
from hunterMakesPy.filesystemToolkit import importLogicalPath2Identifier
from mapFolding.kitAST import IfThis
from mapFolding.kitAST.dataclasses import (
	DeReConstructField2ast, findDataclass, removeDataclass, shatterDataclass, ShatteredDataclass, toFieldsToCallToDataclass)
from mapFolding.kitAST.kitMakeModules import getLogicalPath, getPathFilename
from mapFolding.kitAST.numba.kitNumba import decorateCallableWithNumba, ParametersNumba, parametersNumbaLight
from mapFolding.kitAST.theSSOT import default, defaultMapFolding
from mapFolding.theSSOT import settingsPackage
from typing import TYPE_CHECKING
import ast
import dataclasses
import operator

if TYPE_CHECKING:
	from collections.abc import Sequence
	from mapFolding.theTypes import Default
	from os import PathLike
	from pathlib import PurePath
	from typing import Any

def makeInlineNumba(astModule: ast.Module, identifiers: Default | None = None, **override: Any) -> PurePath:
	"""Generate Numba-optimized sequential implementation of an algorithm."""
	identifiers = identifiers or default
	名CallableSource: str = override.get('名CallableSource') or identifiers['function']['counting']
	名Callable: str = override.get('名Callable') or identifiers['function'].get('inlineNumba') or 名CallableSource
	名CallableDispatcherSource: str | None = override.get('名CallableDispatcherSource') or identifiers['function'].get('dispatcher')
	parametersNumba: ParametersNumba = override.get('parametersNumba') or parametersNumbaLight

	ingredientsFunction = IngredientsFunction(inlineFunctionDef(名CallableSource, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = 名Callable

	shatteredDataclass: ShatteredDataclass = shatterDataclass(*findDataclass(ingredientsFunction))

	ingredientsFunction.imports.update(shatteredDataclass.imports)
	ingredientsFunction: IngredientsFunction = removeDataclass(ingredientsFunction, shatteredDataclass)
	ingredientsFunction.removeUnusedParameters()
	ingredientsFunction = decorateCallableWithNumba(ingredientsFunction, parametersNumba)

	ingredientsModule = IngredientsModule(ingredientsFunction)

	if 名CallableDispatcherSource is not None:

		ingredientsFunctionDispatcher: IngredientsFunction = astModuleToIngredientsFunction(astModule, 名CallableDispatcherSource)
		ingredientsFunctionDispatcher.imports.update(shatteredDataclass.imports)
		名CallableTarget = ingredientsFunction.astFunctionDef.name
		NodeChanger(
			findThis=Be.Call.funcIs(Be.Name.idIs(IfThis.isIdentifier(名CallableSource)))
			, doThat=Grab.funcAttribute(Grab.idAttribute(Then.replaceWith(名CallableTarget)))
		).visit(ingredientsFunctionDispatcher.astFunctionDef)
		ingredientsFunctionDispatcher = toFieldsToCallToDataclass(ingredientsFunctionDispatcher, 名CallableTarget, shatteredDataclass)
		astTuple: ast.Tuple = raiseIfNone(NodeTourist[ast.Return, ast.Tuple](Be.Return.valueIs(Be.Tuple), doThat=Then.extractIt(DOT.value)).captureLastMatch(ingredientsFunction.astFunctionDef))
		astTuple.ctx = Make.Store()

		changeAssignCallToTarget = NodeChanger(
			findThis=Be.Assign.valueIs(IfThis.isCallIdentifier(名CallableTarget))
			, doThat=Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(名CallableTarget), astTuple.elts))))
		changeAssignCallToTarget.visit(ingredientsFunctionDispatcher.astFunctionDef)

		ingredientsModule.appendIngredientsFunction(ingredientsFunctionDispatcher)

	ingredientsModule.removeImportFromModule('numpy')

	名Module: str = override.get('名Module') or identifiers['module']['inlineNumba']
	return toDisk(ingredientsModule, identifiers, override, 名Module)

def makeTheorem2(astModule: ast.Module, identifiers: Default | None = None, **override: Any) -> PurePath:
	"""Generate module by applying optimization predicted by Theorem 2."""
	identifiers = identifiers or default
	名CallableSource: str = override.get('名CallableSource') or identifiers['function']['counting']
	名Callable: str = override.get('名Callable') or identifiers['function'].get('counting') or 名CallableSource
	名Counting: str = override.get('名Counting') or identifiers['variable']['counting']

	ingredientsFunction = IngredientsFunction(inlineFunctionDef(名CallableSource, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = 名Callable

	名dataclassInstance: str = raiseIfNone(NodeTourist[ast.arg, str](Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))

	doubleTheCount: ast.AugAssign = Make.AugAssign(Make.Attribute(Make.Name(名dataclassInstance), 名Counting), Make.Mult(), Make.Constant(2))

	findThisWhile0 = IfThis.isWhileAttributeNamespaceIdentifierGreaterThan0(名dataclassInstance, 'leaf1ndex')
	findThisIf0 = IfThis.isIfAttributeNamespaceIdentifierGreaterThan0(名dataclassInstance, 'leaf1ndex')

	findThis = Be.While.orelseIs(lambda ImaList: ImaList)
	doThat = Grab.orelseAttribute(Grab.index(0, Then.insertThisBelow([doubleTheCount])))
	changer = NodeChanger(findThis, doThat).visit
	findThis = findThisWhile0
	doThat = changer
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef)

	findThis = Be.While.orelseIs(operator.not_)
	doThat = Grab.orelseAttribute(Then.replaceWith([doubleTheCount]))
	changer = NodeChanger(findThis, doThat).visit
	findThis = findThisWhile0
	doThat = changer
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef)

	NodeChanger(
		findThis=findThisWhile0
		, doThat=Grab.testAttribute(Grab.comparatorsAttribute(Then.replaceWith([Make.Constant(4)])))
	).visit(ingredientsFunction.astFunctionDef)

	insertLeaf = NodeTourist[ast.If, list[ast.stmt]](
		findThis=findThisIf0
		, doThat=Then.extractIt(DOT.body)
	).captureLastMatch(ingredientsFunction.astFunctionDef)
	NodeChanger(
		findThis=findThisIf0
		, doThat=Then.replaceWith(insertLeaf)
	).visit(ingredientsFunction.astFunctionDef)

	findThis_leftIsDOTleaf1ndex = Be.Compare.leftIs(IfThis.isAttributeNamespaceIdentifier(名dataclassInstance, 'leaf1ndex'))
	findThis_comparatorsIs0 = Be.Compare.comparatorsIs(Be.at(0, IfThis.isConstant_value(0)))

#========== isAttributeNamespaceIdentifierGreaterThan0 ======
	findThis = findThis_leftIsDOTleaf1ndex
	doThat = NodeChanger(Be.Compare.opsIs(Be.at(0, Be.Gt)), NodeChanger(findThis_comparatorsIs0, Then.removeIt).visit).visit
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef)

#========== isAttributeNamespaceIdentifierLessThanOrEqual0 ======
	findThis = findThis_leftIsDOTleaf1ndex
	doThat = NodeChanger(Be.Compare.opsIs(Be.at(0, Be.LtE)), Then.removeIt).visit
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef)

	ingredientsModule = IngredientsModule(ingredientsFunction)
	del ingredientsFunction

	logicalPathInfix: identifierDotAttribute = override.get('logicalPathInfix') or identifiers['logicalPath']['synthetic']
	名Package: str = override.get('package') or identifiers['module']['package']

	名CallableDispatcherSource: str | None = override.get('名CallableDispatcherSource') or identifiers['function'].get('dispatcher')
	# DEVELOPMENT When `名CallableDispatcherSource` was a parameter, passing it signalled to modify the dispatcher.
	if 名CallableDispatcherSource is not None:
		ingredientsFunction: IngredientsFunction = astModuleToIngredientsFunction(astModule, 名CallableDispatcherSource)

		#Update calls to the original function name with the new function name
		NodeChanger(
			findThis=Be.Call.funcIs(Be.Name.idIs(IfThis.isIdentifier(名CallableSource)))
			, doThat=Grab.funcAttribute(Grab.idAttribute(Then.replaceWith(名Callable)))
		).visit(ingredientsFunction.astFunctionDef)

		名CallableInitializeState: str = override.get('名CallableInitializeState') or identifiers['function']['initializeState']
		AssignInitializedDataclass: ast.Assign = Make.Assign([Make.Name(名dataclassInstance)], value=Make.Call(Make.Name(名CallableInitializeState), [Make.Name(名dataclassInstance)]))

		#Insert the transitionOnGroupsOfFolds call at the beginning of the function
		ingredientsFunction.astFunctionDef.body.insert(0, AssignInitializedDataclass)

		名ModuleInitializeState: str = override.get('名ModuleInitializeState') or identifiers['module']['initializeState']
		dotModule: identifierDotAttribute = getLogicalPath(名Package, logicalPathInfix, 名ModuleInitializeState)
		ingredientsFunction.imports.addImportFrom_asStr(dotModule, 名CallableInitializeState)

		ingredientsModule.appendIngredientsFunction(ingredientsFunction)

	名Module: str = override.get('名Module') or identifiers['module']['theorem2']

	return toDisk(ingredientsModule, identifiers, override, 名Module)

# SEMIOTICS
def numbaOnTheorem2(astModule: ast.Module, identifiers: Default | None = None, **override: Any) -> PurePath:
	"""Generate Numba-accelerated Theorem 2 implementation with dataclass decomposition."""
	identifiers = identifiers or default
	名CallableSource: str = override.get('名CallableSource') or identifiers['function'].get('theorem2Trimmed') or identifiers['function'].get('theorem2') or identifiers['function']['counting']
	名Callable: str = override.get('名Callable') or identifiers['function'].get('theorem2Numba') or 名CallableSource
	名CallableDispatcherSource: str | None = override.get('名CallableDispatcherSource') or identifiers['function'].get('dispatcher')
	parametersNumba: ParametersNumba = override.get('parametersNumba') or parametersNumbaLight

	ingredientsFunction = IngredientsFunction(inlineFunctionDef(名CallableSource, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = 名Callable

	logicalPathDataclass, identifierDataclass, identifierDataclassInstance = findDataclass(ingredientsFunction)

	shatteredDataclass: ShatteredDataclass = shatterDataclass(logicalPathDataclass, identifierDataclass, identifierDataclassInstance)

	ingredientsFunction.imports.update(shatteredDataclass.imports)
	ingredientsFunction: IngredientsFunction = removeDataclass(ingredientsFunction, shatteredDataclass)
	ingredientsFunction.removeUnusedParameters()
	ingredientsFunction = decorateCallableWithNumba(ingredientsFunction, parametersNumba)

	ingredientsModule = IngredientsModule(ingredientsFunction)
	ingredientsModule.removeImportFromModule('numpy')

	if 名CallableDispatcherSource is not None:
		ingredientsFunctionDispatcher: IngredientsFunction = astModuleToIngredientsFunction(astModule, 名CallableDispatcherSource)
		ingredientsFunctionDispatcher.imports.update(shatteredDataclass.imports)
		名CallableTarget = ingredientsFunction.astFunctionDef.name
		NodeChanger(
			findThis=Be.Call.funcIs(Be.Name.idIs(IfThis.isIdentifier(名CallableSource)))
			, doThat=Grab.funcAttribute(Grab.idAttribute(Then.replaceWith(名CallableTarget)))
		).visit(ingredientsFunctionDispatcher.astFunctionDef)
		ingredientsFunctionDispatcher = toFieldsToCallToDataclass(ingredientsFunctionDispatcher, 名CallableTarget, shatteredDataclass)
		astTuple: ast.Tuple = raiseIfNone(NodeTourist[ast.Return, ast.Tuple](Be.Return.valueIs(Be.Tuple), doThat=Then.extractIt(DOT.value)).captureLastMatch(ingredientsFunction.astFunctionDef))
		astTuple.ctx = Make.Store()

		changeAssignCallToTarget = NodeChanger(
			findThis=Be.Assign.valueIs(IfThis.isCallIdentifier(名CallableTarget))
			, doThat=Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(名CallableTarget), astTuple.elts))))
		changeAssignCallToTarget.visit(ingredientsFunctionDispatcher.astFunctionDef)

		ingredientsModule.appendIngredientsFunction(ingredientsFunctionDispatcher)

	ingredientsModule.removeImportFromModule('numpy')

	名Module: str = override.get('名Module') or identifiers['module']['theorem2Numba']

	return toDisk(ingredientsModule, identifiers, override, 名Module)

def trimTheorem2(astModule: ast.Module, identifiers: Default | None = None, **override: Any) -> PurePath:
	"""Generate constrained Theorem 2 implementation by removing unnecessary logic."""
	identifiers = identifiers or default
	名CallableSource: str = override.get('名CallableSource') or identifiers['function'].get('theorem2') or identifiers['function']['counting']
	名Callable: str = override.get('名Callable') or identifiers['function'].get('theorem2Trimmed') or 名CallableSource
	名CallableDispatcherSource: str | None = override.get('名CallableDispatcherSource') or identifiers['function'].get('dispatcher')

	ingredientsFunction = IngredientsFunction(inlineFunctionDef(名CallableSource, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = 名Callable

	identifierDataclassInstance: str = raiseIfNone(NodeTourist[ast.arg, str](Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))

	NodeChanger(
		findThis=IfThis.isIfUnaryNotAttributeNamespaceIdentifier(identifierDataclassInstance, 'dimensionsUnconstrained')
		, doThat=Then.removeIt
	).visit(ingredientsFunction.astFunctionDef)

	ingredientsModule = IngredientsModule(ingredientsFunction)
	ingredientsModule.removeImportFromModule('numpy')

	if 名CallableDispatcherSource is not None:
		ingredientsFunctionDispatcher: IngredientsFunction = astModuleToIngredientsFunction(astModule, 名CallableDispatcherSource)
		名CallableTarget = ingredientsFunction.astFunctionDef.name

		#Update any calls to the original function name with the new target function name
		NodeChanger(
			findThis=Be.Call.funcIs(Be.Name.idIs(IfThis.isIdentifier(名CallableSource)))
			, doThat=Grab.funcAttribute(Grab.idAttribute(Then.replaceWith(名CallableTarget)))
		).visit(ingredientsFunctionDispatcher.astFunctionDef)

		ingredientsModule.appendIngredientsFunction(ingredientsFunctionDispatcher)

	名Module: str = override.get('名Module') or identifiers['module']['theorem2Trimmed']

	return toDisk(ingredientsModule, identifiers, override, 名Module)

def makeInlineParallelNumba(astModule: ast.Module, identifiers: Default | None = None, **override: Any) -> PurePath:
	"""Generate parallel implementation with concurrent execution and task division."""
	identifiers = identifiers or defaultMapFolding
	名CallableSource: str = override.get('名CallableSource') or identifiers['function']['counting']
	名Callable: str = override.get('名Callable') or identifiers['function'].get('inlineParallelNumba') or 名CallableSource
	名CallableDispatcherSource: str = override.get('名CallableDispatcherSource') or identifiers['function']['dispatcher']
	名CallableDispatcher: str = override.get('名CallableDispatcher') or identifiers['function'].get('inlineParallelDispatcher') or 名CallableDispatcherSource
	名Counting: str = override.get('名Counting') or identifiers['variable']['counting']
	parametersNumba: ParametersNumba = override.get('parametersNumba') or parametersNumbaLight

	ingredientsFunction = IngredientsFunction(inlineFunctionDef(名CallableSource, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = 名Callable

	logicalPathDataclass, identifierDataclass, identifierDataclassInstance = findDataclass(ingredientsFunction)

	shatteredDataclass: ShatteredDataclass = shatterDataclass(logicalPathDataclass, identifierDataclass, identifierDataclassInstance)

#-START add the parallel state fields to the count function ------------------------------------------------
	dataclassBaseFields: tuple[dataclasses.Field[Any], ...] = dataclasses.fields(importLogicalPath2Identifier(logicalPathDataclass, identifierDataclass))
	名dataclassParallel: identifierDotAttribute = 'Parallel' + identifierDataclass
	dataclassFieldsParallel: tuple[dataclasses.Field[Any], ...] = dataclasses.fields(importLogicalPath2Identifier(logicalPathDataclass, 名dataclassParallel))
	onlyParallelFields: list[dataclasses.Field[Any]] = [field for field in dataclassFieldsParallel if field.name not in [fieldBase.name for fieldBase in dataclassBaseFields]]

	Official_fieldOrder: list[str] = []
	dictionaryDeReConstruction: dict[str, DeReConstructField2ast] = {}

	dataclassClassDef: ast.ClassDef | None = extractClassDef(parseLogicalPath2astModule(logicalPathDataclass), 名dataclassParallel)
	if not dataclassClassDef:
		message = f"I could not find `{名dataclassParallel = }` in `{logicalPathDataclass = }`."
		raise ValueError(message)

	for aField in onlyParallelFields:
		Official_fieldOrder.append(aField.name)
		dictionaryDeReConstruction[aField.name] = DeReConstructField2ast(logicalPathDataclass, dataclassClassDef, identifierDataclassInstance, aField)

	shatteredDataclassParallel = ShatteredDataclass(
		countingVariableAnnotation=shatteredDataclass.countingVariableAnnotation,
		countingVariableName=shatteredDataclass.countingVariableName,
		field2AnnAssign={**shatteredDataclass.field2AnnAssign, **{dictionaryDeReConstruction[field].name: dictionaryDeReConstruction[field].astAnnAssignConstructor for field in Official_fieldOrder}},
		Z0Z_field2AnnAssign={**shatteredDataclass.Z0Z_field2AnnAssign, **{dictionaryDeReConstruction[field].name: dictionaryDeReConstruction[field].Z0Z_hack for field in Official_fieldOrder}},
		boxOf_argAnnotated4ArgumentsSpecification=shatteredDataclass.boxOf_argAnnotated4ArgumentsSpecification + [dictionaryDeReConstruction[field].ast_argAnnotated for field in Official_fieldOrder],
		boxOf_keyword_field__field4init=shatteredDataclass.boxOf_keyword_field__field4init + [dictionaryDeReConstruction[field].ast_keyword_field__field for field in Official_fieldOrder if dictionaryDeReConstruction[field].init],
		boxOfAnnotations=shatteredDataclass.boxOfAnnotations + [dictionaryDeReConstruction[field].astAnnotation for field in Official_fieldOrder],
		boxOfName4Parameters=shatteredDataclass.boxOfName4Parameters + [dictionaryDeReConstruction[field].astName for field in Official_fieldOrder],
		boxOfUnpack=shatteredDataclass.boxOfUnpack + [Make.AnnAssign(dictionaryDeReConstruction[field].astName, dictionaryDeReConstruction[field].astAnnotation, dictionaryDeReConstruction[field].ast_nameDOTname) for field in Official_fieldOrder],
		map_stateDOTfield2Name={**shatteredDataclass.map_stateDOTfield2Name, **{dictionaryDeReConstruction[field].ast_nameDOTname: dictionaryDeReConstruction[field].astName for field in Official_fieldOrder}},
		)
	shatteredDataclassParallel.fragments4AssignmentOrParameters = Make.Tuple(shatteredDataclassParallel.boxOfName4Parameters, Make.Store())
	shatteredDataclassParallel.repack = Make.Assign([Make.Name(identifierDataclassInstance)], value=Make.Call(Make.Name(名dataclassParallel), list_keyword=shatteredDataclassParallel.boxOf_keyword_field__field4init))
	shatteredDataclassParallel.signatureReturnAnnotation = Make.Subscript(Make.Name('tuple'), Make.Tuple(shatteredDataclassParallel.boxOfAnnotations))

	shatteredDataclassParallel.imports.update(*(dictionaryDeReConstruction[field].ledger for field in Official_fieldOrder))
	shatteredDataclassParallel.imports.addImportFrom_asStr(logicalPathDataclass, 名dataclassParallel)
	shatteredDataclassParallel.imports.update(shatteredDataclass.imports)
	shatteredDataclassParallel.imports.removeImportFrom(logicalPathDataclass, identifierDataclass)

#-END add the parallel state fields to the count function ------------------------------------------------

	ingredientsFunction.imports.update(shatteredDataclassParallel.imports)
	ingredientsFunction: IngredientsFunction = removeDataclass(ingredientsFunction, shatteredDataclassParallel)

#-START add the parallel logic to the count function ------------------------------------------------

	findThis = Be.While.testIs(Be.Compare.leftIs(IfThis.isNameIdentifier('leafConnectee')))
	captureCountGapsCodeBlock: NodeTourist[ast.While, Sequence[ast.stmt]] = NodeTourist(findThis, doThat=Then.extractIt(DOT.body))
	countGapsCodeBlock: Sequence[ast.stmt] = raiseIfNone(captureCountGapsCodeBlock.captureLastMatch(ingredientsFunction.astFunctionDef))

	thisIsMyTaskIndexCodeBlock = Make.If(Make.Or.join([Make.Compare(Make.Name('leaf1ndex'), ops=[Make.NotEq()], comparators=[Make.Name('taskDivisions')])
				, Make.Compare(Make.Mod.join([Make.Name('leafConnectee'), Make.Name('taskDivisions')]), ops=[Make.Eq()], comparators=[Make.Name('task次')])
			]), body=list(countGapsCodeBlock[0:-1]))

	countGapsCodeBlockNew: list[ast.stmt] = [thisIsMyTaskIndexCodeBlock, countGapsCodeBlock[-1]]
	NodeChanger[ast.While, hasDOTbody](findThis, doThat=Grab.bodyAttribute(Then.replaceWith(countGapsCodeBlockNew))).visit(ingredientsFunction.astFunctionDef)

#-END add the parallel logic to the count function ------------------------------------------------

	ingredientsFunction.removeUnusedParameters()

	ingredientsFunction = decorateCallableWithNumba(ingredientsFunction, parametersNumba)

#-START unpack/repack the dataclass function ------------------------------------------------
	unRepackDataclass: IngredientsFunction = astModuleToIngredientsFunction(astModule, 名CallableDispatcherSource)
	unRepackDataclass.astFunctionDef.name = 'unRepack' + 名dataclassParallel
	unRepackDataclass.imports.update(shatteredDataclassParallel.imports)
	NodeChanger(
			findThis=Be.arg.annotationIs(Be.Name.idIs(lambda thisAttribute: thisAttribute == identifierDataclass))
			, doThat=Grab.annotationAttribute(Grab.idAttribute(Then.replaceWith(名dataclassParallel)))
		).visit(unRepackDataclass.astFunctionDef)
	unRepackDataclass.astFunctionDef.returns = Make.Name(名dataclassParallel)
	名CallableTarget: identifierDotAttribute = ingredientsFunction.astFunctionDef.name
	NodeChanger(
		findThis=Be.Call.funcIs(Be.Name.idIs(IfThis.isIdentifier(名CallableSource)))
		, doThat=Grab.funcAttribute(Grab.idAttribute(Then.replaceWith(名CallableTarget)))
	).visit(unRepackDataclass.astFunctionDef)
	unRepackDataclass = toFieldsToCallToDataclass(unRepackDataclass, 名CallableTarget, shatteredDataclassParallel)

	astTuple: ast.Tuple = raiseIfNone(NodeTourist[ast.Return, ast.Tuple | None](Be.Return, Then.extractIt(DOT.value)).captureLastMatch(ingredientsFunction.astFunctionDef))
	astTuple.ctx = Make.Store()
	changeAssignCallToTarget: NodeChanger[ast.Assign, ast.Assign] = NodeChanger(
		findThis=Be.Assign.valueIs(IfThis.isCallIdentifier(名CallableTarget))
		, doThat=Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(名CallableTarget), astTuple.elts)))
	)
	changeAssignCallToTarget.visit(unRepackDataclass.astFunctionDef)

	ingredientsDoTheNeedful: IngredientsFunction = IngredientsFunction(
		astFunctionDef=Make.FunctionDef(名CallableDispatcher
			, argumentSpecification=Make.arguments(list_arg=[Make.arg(identifierDataclassInstance, annotation=Make.Name(名dataclassParallel)), Make.arg('concurrencyLimit', annotation=Make.Name('int'))])
			, body=[Make.Assign([Make.Name('stateParallel', Make.Store())], value=Make.Call(Make.Name('deepcopy'), listParameters=[Make.Name(identifierDataclassInstance)]))
				, Make.AnnAssign(Make.Name('boxOfStatesParallel', Make.Store()), annotation=Make.Subscript(value=Make.Name('list'), slice=Make.Name(名dataclassParallel))
					, value=Make.Mult.join([Make.List([Make.Name('stateParallel')]), Make.Attribute(Make.Name('stateParallel'), 'taskDivisions')]))
				, Make.AnnAssign(Make.Name('groupsOfTotalFolds', Make.Store()), annotation=Make.Name('int'), value=Make.Constant(value=0))

				, Make.AnnAssign(Make.Name('dictionaryConcurrency', Make.Store()), annotation=Make.Subscript(value=Make.Name('dict'), slice=Make.Tuple([Make.Name('int'), Make.Subscript(value=Make.Name('ConcurrentFuture'), slice=Make.Name(名dataclassParallel))])), value=Make.Dict())
				, Make.With(items=[Make.withitem(context_expr=Make.Call(Make.Name('ProcessPoolExecutor'), listParameters=[Make.Name('concurrencyLimit')]), optional_vars=Make.Name('concurrencyManager', Make.Store()))]
					, body=[Make.For(Make.Name('indexSherpa', Make.Store()), iter=Make.Call(Make.Name('range'), listParameters=[Make.Attribute(Make.Name('stateParallel'), 'taskDivisions')])
							, body=[Make.Assign([Make.Name(identifierDataclassInstance, Make.Store())], value=Make.Call(Make.Name('deepcopy'), listParameters=[Make.Name('stateParallel')]))
								, Make.Assign([Make.Attribute(Make.Name(identifierDataclassInstance), 'task次', context=Make.Store())], value=Make.Name('indexSherpa'))
								, Make.Assign([Make.Subscript(Make.Name('dictionaryConcurrency'), slice=Make.Name('indexSherpa'), context=Make.Store())], value=Make.Call(Make.Attribute(Make.Name('concurrencyManager'), 'submit'), listParameters=[Make.Name(unRepackDataclass.astFunctionDef.name), Make.Name(identifierDataclassInstance)]))])
						, Make.For(Make.Name('indexSherpa', Make.Store()), iter=Make.Call(Make.Name('range'), listParameters=[Make.Attribute(Make.Name('stateParallel'), 'taskDivisions')])
							, body=[Make.Assign([Make.Subscript(Make.Name('boxOfStatesParallel'), slice=Make.Name('indexSherpa'), context=Make.Store())], value=Make.Call(Make.Attribute(Make.Subscript(Make.Name('dictionaryConcurrency'), slice=Make.Name('indexSherpa')), 'result')))
								, Make.AugAssign(Make.Name('groupsOfTotalFolds', Make.Store()), op=Make.Add(), value=Make.Attribute(Make.Subscript(Make.Name('boxOfStatesParallel'), slice=Make.Name('indexSherpa')), 名Counting))])])

				, Make.AnnAssign(Make.Name('totalFolds', Make.Store()), annotation=Make.Name('int'), value=Make.Mult.join([Make.Name('groupsOfTotalFolds'), Make.Attribute(Make.Name('stateParallel'), 'totalLeaves')]))
				, Make.Return(Make.Tuple([Make.Name('totalFolds'), Make.Name('boxOfStatesParallel')]))]
			, returns=Make.Subscript(Make.Name('tuple'), slice=Make.Tuple([Make.Name('int'), Make.Subscript(Make.Name('list'), slice=Make.Name(名dataclassParallel))])))
		, imports=LedgerOfImports(Make.Module([Make.ImportFrom('concurrent.futures', list_alias=[Make.alias('Future', asName='ConcurrentFuture'), Make.alias('ProcessPoolExecutor')]),
			Make.ImportFrom('copy', list_alias=[Make.alias('deepcopy')]),
			Make.ImportFrom('multiprocessing', list_alias=[Make.alias('set_start_method', asName='multiprocessing_set_start_method')])])
		)
	)

	ingredientsModule = IngredientsModule([ingredientsFunction, unRepackDataclass, ingredientsDoTheNeedful]
						, prologue=Make.Module([Make.If(test=Make.Compare(left=Make.Name('__name__'), ops=[Make.Eq()], comparators=[Make.Constant('__main__')]), body=[Make.Expr(Make.Call(Make.Name('multiprocessing_set_start_method'), listParameters=[Make.Constant('spawn')]))])])
	)
	ingredientsModule.removeImportFromModule('numpy')
	名Module: str = override.get('名Module') or identifiers['module']['countParallelNumba']

	return toDisk(ingredientsModule, identifiers, override, 名Module)

def toDisk(ingredientsModule: IngredientsModule, identifiers: Default, keywords: dict[str, Any], identifierModule: str, **override: Any) -> PurePath:
	"""Write a generated module to its configured output path.

	(AI generated docstring)

	You can use this function to resolve the output location for `ingredientsModule` from
	`identifiers`, `keywords`, and `override`, then write the generated module to disk. The
	function applies explicit overrides before identifier defaults and returns the final path.

	Parameters
	----------
	ingredientsModule : IngredientsModule
		Generated module wrapper that knows how to write the assembled module.
	identifiers : Default
		Default identifier mapping that provides package, path, and logical-path fallbacks.
	keywords : dict[str, Any]
		Keyword overrides forwarded from the caller.
	identifierModule : str
		Module identifier used when constructing the destination filename.
	**override : Any
		Explicit override values that take precedence over `keywords` and `identifiers`.

	Returns
	-------
	pathFilename : PurePath
		Path to the written module file.
	"""
	logicalPathInfix: identifierDotAttribute = override.get('logicalPathInfix') or keywords.get('logicalPathInfix') or identifiers['logicalPath']['synthetic']
	pathRoot: PathLike[str] = override.get('pathRoot') or keywords.get('pathRoot') or identifiers['filesystem']['pathRoot']
	identifierPackage: str = override.get('package') or keywords.get('package') or identifiers['module']['package']
	fileExtension: str = override.get('fileExtension') or settingsPackage.fileExtension
	pathFilename: PurePath = override.get('pathFilename') or getPathFilename(pathRoot, logicalPathInfix, identifierModule, fileExtension)
	return ingredientsModule.write_astModule(pathFilename, identifierPackage)
