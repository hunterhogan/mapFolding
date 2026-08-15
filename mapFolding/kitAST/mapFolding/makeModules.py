"""makeMapFoldingModules."""
from __future__ import annotations

from astToolkit import (
	Be, DOT, extractClassDef, Grab, hasDOTbody, identifierDotAttribute, Make, NodeChanger, NodeTourist, parseLogicalPath2astModule,
	parsePathFilename2astModule, Then)
from astToolkit.containers import astModuleToIngredientsFunction, IngredientsFunction, IngredientsModule, LedgerOfImports
from astToolkit.transformationTools import inlineFunctionDef
from hunterMakesPy import raiseIfNone
from hunterMakesPy.filesystemToolkit import importLogicalPath2Identifier
from mapFolding.kitAST import DeReConstructField2ast, IfThis, ShatteredDataclass
from mapFolding.kitAST.kitMakeModules import findDataclass, getModule, getPathFilename
from mapFolding.kitAST.kitTransformations import (
	removeDataclassFromFunction, shatter_dataclassesDOTdataclass, unpackDataclassCallFunctionRepackDataclass)
from mapFolding.kitAST.mapFolding.makeModules_count import (
	makeDaoOfMapFoldingNumba, makeInlineNumba, makeTheorem2, numbaOnTheorem2, trimTheorem2)
from mapFolding.kitAST.mapFolding.makeModules_doTheNeedful import makeInitializeState
from mapFolding.kitAST.numba.kitNumba import decorateCallableWithNumba, parametersNumbaLight
from mapFolding.kitAST.theSSOT import defaultMapFolding
from mapFolding.theSSOT import settingsPackage
from typing import TYPE_CHECKING
import ast
import dataclasses

if TYPE_CHECKING:
	from collections.abc import Sequence
	from pathlib import PurePath
	from typing import Any

def makeInlineParallelNumba(astModule: ast.Module, identifierModule: str, identifierCallable: str | None = None, logicalPathInfix: identifierDotAttribute | None = None, _sourceCallableDispatcher: str | None = None) -> PurePath:
	"""Generate parallel implementation with concurrent execution and task division.

	Parameters
	----------
	astModule : ast.Module
		Source module containing the base algorithm.
	identifierModule : str
		Name for the generated parallel module.
	identifierCallable : str | None = None
		Name for the core parallel counting function.
	logicalPathInfix : identifierDotAttribute | None = None
		Directory path for organizing the generated module.
	_sourceCallableDispatcher : str | None = None
		Optional dispatcher function identifier.

	Returns
	-------
	pathFilename : PurePath
		Filesystem path where the parallel module was written.

	"""
	sourceCallableIdentifier = defaultMapFolding['function']['counting']
	if identifierCallable is None:
		identifierCallable = sourceCallableIdentifier
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = identifierCallable

	logicalPathDataclass, identifierDataclass, identifierDataclassInstance = findDataclass(ingredientsFunction)

	shatteredDataclass: ShatteredDataclass = shatter_dataclassesDOTdataclass(logicalPathDataclass, identifierDataclass, identifierDataclassInstance)

#-START add the parallel state fields to the count function ------------------------------------------------
	dataclassBaseFields: tuple[dataclasses.Field[Any], ...] = dataclasses.fields(importLogicalPath2Identifier(logicalPathDataclass, identifierDataclass))
	dataclassIdentifierParallel: identifierDotAttribute = 'Parallel' + identifierDataclass
	dataclassFieldsParallel: tuple[dataclasses.Field[Any], ...] = dataclasses.fields(importLogicalPath2Identifier(logicalPathDataclass, dataclassIdentifierParallel))
	onlyParallelFields: list[dataclasses.Field[Any]] = [field for field in dataclassFieldsParallel if field.name not in [fieldBase.name for fieldBase in dataclassBaseFields]]

	Official_fieldOrder: list[str] = []
	dictionaryDeReConstruction: dict[str, DeReConstructField2ast] = {}

	dataclassClassDef: ast.ClassDef | None = extractClassDef(parseLogicalPath2astModule(logicalPathDataclass), dataclassIdentifierParallel)
	if not dataclassClassDef:
		message = f"I could not find `{dataclassIdentifierParallel = }` in `{logicalPathDataclass = }`."
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
	shatteredDataclassParallel.repack = Make.Assign([Make.Name(identifierDataclassInstance)], value=Make.Call(Make.Name(dataclassIdentifierParallel), list_keyword=shatteredDataclassParallel.boxOf_keyword_field__field4init))
	shatteredDataclassParallel.signatureReturnAnnotation = Make.Subscript(Make.Name('tuple'), Make.Tuple(shatteredDataclassParallel.boxOfAnnotations))

	shatteredDataclassParallel.imports.update(*(dictionaryDeReConstruction[field].ledger for field in Official_fieldOrder))
	shatteredDataclassParallel.imports.addImportFrom_asStr(logicalPathDataclass, dataclassIdentifierParallel)
	shatteredDataclassParallel.imports.update(shatteredDataclass.imports)
	shatteredDataclassParallel.imports.removeImportFrom(logicalPathDataclass, identifierDataclass)

#-END add the parallel state fields to the count function ------------------------------------------------

	ingredientsFunction.imports.update(shatteredDataclassParallel.imports)
	ingredientsFunction: IngredientsFunction = removeDataclassFromFunction(ingredientsFunction, shatteredDataclassParallel)

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

	ingredientsFunction = decorateCallableWithNumba(ingredientsFunction, parametersNumbaLight)

#-START unpack/repack the dataclass function ------------------------------------------------
	sourceCallableIdentifier = defaultMapFolding['function']['dispatcher']

	unRepackDataclass: IngredientsFunction = astModuleToIngredientsFunction(astModule, sourceCallableIdentifier)
	unRepackDataclass.astFunctionDef.name = 'unRepack' + dataclassIdentifierParallel
	unRepackDataclass.imports.update(shatteredDataclassParallel.imports)
	NodeChanger(
			findThis=Be.arg.annotationIs(Be.Name.idIs(lambda thisAttribute: thisAttribute == identifierDataclass))
			, doThat=Grab.annotationAttribute(Grab.idAttribute(Then.replaceWith(dataclassIdentifierParallel)))
		).visit(unRepackDataclass.astFunctionDef)
	unRepackDataclass.astFunctionDef.returns = Make.Name(dataclassIdentifierParallel)
	targetCallableIdentifier: identifierDotAttribute = ingredientsFunction.astFunctionDef.name
	unRepackDataclass = unpackDataclassCallFunctionRepackDataclass(unRepackDataclass, targetCallableIdentifier, shatteredDataclassParallel)

	astTuple: ast.Tuple = raiseIfNone(NodeTourist[ast.Return, ast.Tuple | None](Be.Return, Then.extractIt(DOT.value)).captureLastMatch(ingredientsFunction.astFunctionDef))
	astTuple.ctx = Make.Store()
	changeAssignCallToTarget: NodeChanger[ast.Assign, ast.Assign] = NodeChanger(
		findThis=Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier))
		, doThat=Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(targetCallableIdentifier), astTuple.elts)))
	)
	changeAssignCallToTarget.visit(unRepackDataclass.astFunctionDef)

	ingredientsDoTheNeedful: IngredientsFunction = IngredientsFunction(
		astFunctionDef=Make.FunctionDef('doTheNeedful'
			, argumentSpecification=Make.arguments(list_arg=[Make.arg('state', annotation=Make.Name(dataclassIdentifierParallel)), Make.arg('concurrencyLimit', annotation=Make.Name('int'))])
			, body=[Make.Assign([Make.Name('stateParallel', Make.Store())], value=Make.Call(Make.Name('deepcopy'), listParameters=[Make.Name('state')]))
				, Make.AnnAssign(Make.Name('boxOfStatesParallel', Make.Store()), annotation=Make.Subscript(value=Make.Name('list'), slice=Make.Name(dataclassIdentifierParallel))
					, value=Make.Mult.join([Make.List([Make.Name('stateParallel')]), Make.Attribute(Make.Name('stateParallel'), 'taskDivisions')]))
				, Make.AnnAssign(Make.Name('groupsOfTotalFolds', Make.Store()), annotation=Make.Name('int'), value=Make.Constant(value=0))

				, Make.AnnAssign(Make.Name('dictionaryConcurrency', Make.Store()), annotation=Make.Subscript(value=Make.Name('dict'), slice=Make.Tuple([Make.Name('int'), Make.Subscript(value=Make.Name('ConcurrentFuture'), slice=Make.Name(dataclassIdentifierParallel))])), value=Make.Dict())
				, Make.With(items=[Make.withitem(context_expr=Make.Call(Make.Name('ProcessPoolExecutor'), listParameters=[Make.Name('concurrencyLimit')]), optional_vars=Make.Name('concurrencyManager', Make.Store()))]
					, body=[Make.For(Make.Name('indexSherpa', Make.Store()), iter=Make.Call(Make.Name('range'), listParameters=[Make.Attribute(Make.Name('stateParallel'), 'taskDivisions')])
							, body=[Make.Assign([Make.Name('state', Make.Store())], value=Make.Call(Make.Name('deepcopy'), listParameters=[Make.Name('stateParallel')]))
								, Make.Assign([Make.Attribute(Make.Name('state'), 'task次', context=Make.Store())], value=Make.Name('indexSherpa'))
								, Make.Assign([Make.Subscript(Make.Name('dictionaryConcurrency'), slice=Make.Name('indexSherpa'), context=Make.Store())], value=Make.Call(Make.Attribute(Make.Name('concurrencyManager'), 'submit'), listParameters=[Make.Name(unRepackDataclass.astFunctionDef.name), Make.Name('state')]))])
						, Make.For(Make.Name('indexSherpa', Make.Store()), iter=Make.Call(Make.Name('range'), listParameters=[Make.Attribute(Make.Name('stateParallel'), 'taskDivisions')])
							, body=[Make.Assign([Make.Subscript(Make.Name('boxOfStatesParallel'), slice=Make.Name('indexSherpa'), context=Make.Store())], value=Make.Call(Make.Attribute(Make.Subscript(Make.Name('dictionaryConcurrency'), slice=Make.Name('indexSherpa')), 'result')))
								, Make.AugAssign(Make.Name('groupsOfTotalFolds', Make.Store()), op=Make.Add(), value=Make.Attribute(Make.Subscript(Make.Name('boxOfStatesParallel'), slice=Make.Name('indexSherpa')), 'groupsOfFolds'))])])

				, Make.AnnAssign(Make.Name('totalFolds', Make.Store()), annotation=Make.Name('int'), value=Make.Mult.join([Make.Name('groupsOfTotalFolds'), Make.Attribute(Make.Name('stateParallel'), 'totalLeaves')]))
				, Make.Return(Make.Tuple([Make.Name('totalFolds'), Make.Name('boxOfStatesParallel')]))]
			, returns=Make.Subscript(Make.Name('tuple'), slice=Make.Tuple([Make.Name('int'), Make.Subscript(Make.Name('list'), slice=Make.Name(dataclassIdentifierParallel))])))
		, imports=LedgerOfImports(Make.Module([Make.ImportFrom('concurrent.futures', list_alias=[Make.alias('Future', asName='ConcurrentFuture'), Make.alias('ProcessPoolExecutor')]),
			Make.ImportFrom('copy', list_alias=[Make.alias('deepcopy')]),
			Make.ImportFrom('multiprocessing', list_alias=[Make.alias('set_start_method', asName='multiprocessing_set_start_method')])])
		)
	)

	ingredientsModule = IngredientsModule([ingredientsFunction, unRepackDataclass, ingredientsDoTheNeedful]
						, prologue=Make.Module([Make.If(test=Make.Compare(left=Make.Name('__name__'), ops=[Make.Eq()], comparators=[Make.Constant('__main__')]), body=[Make.Expr(Make.Call(Make.Name('multiprocessing_set_start_method'), listParameters=[Make.Constant('spawn')]))])])
	)
	ingredientsModule.removeImportFromModule('numpy')

	pathFilename: PurePath = getPathFilename(settingsPackage.pathPackage, logicalPathInfix, identifierModule)

	ingredientsModule.write_astModule(pathFilename, settingsPackage.identifierPackage)

	return pathFilename

def makeModulesMapFolding() -> None:
	"""Make multidimensional map folding modules."""
	astModule = getModule(logicalPathInfix='algorithms', identifierModule=defaultMapFolding['module']['algorithm'])
	pathFilename: PurePath = makeDaoOfMapFoldingNumba(astModule, 'daoOfMapFoldingNumba', None, defaultMapFolding['logicalPath']['synthetic'], defaultMapFolding['function']['dispatcher'])

	astModule = getModule(logicalPathInfix='algorithms', identifierModule=defaultMapFolding['module']['algorithm'])
	pathFilename = makeInlineNumba(astModule, 'inlineNumba', None, defaultMapFolding['logicalPath']['synthetic'], defaultMapFolding['function']['dispatcher'])

	astModule = getModule(logicalPathInfix='algorithms', identifierModule=defaultMapFolding['module']['algorithm'])
	pathFilename = makeInlineParallelNumba(astModule, 'countParallelNumba', None, defaultMapFolding['logicalPath']['synthetic'], defaultMapFolding['function']['dispatcher'])

	astModule: ast.Module = getModule(logicalPathInfix='algorithms', identifierModule=defaultMapFolding['module']['algorithm'])
	makeInitializeState(astModule, defaultMapFolding['module']['initializeState'], defaultMapFolding['function']['initializeState'], defaultMapFolding['logicalPath']['synthetic'], identifiers=defaultMapFolding)

	astModule = getModule(logicalPathInfix='algorithms', identifierModule=defaultMapFolding['module']['algorithm'])
	pathFilename = makeTheorem2(astModule, 'theorem2', None, defaultMapFolding['logicalPath']['synthetic'], defaultMapFolding['function']['dispatcher'], identifiers=defaultMapFolding)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = trimTheorem2(astModule, 'theorem2Trimmed', None, defaultMapFolding['logicalPath']['synthetic'], defaultMapFolding['function']['dispatcher'])

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = numbaOnTheorem2(astModule, 'theorem2Numba', None, defaultMapFolding['logicalPath']['synthetic'], defaultMapFolding['function']['dispatcher'])

if __name__ == '__main__':
	makeModulesMapFolding()
