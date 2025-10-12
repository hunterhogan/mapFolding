"""makeMapFoldingModules."""
from astToolkit import (
	Be, DOT, extractClassDef, Grab, hasDOTbody, identifierDotAttribute, Make, NodeChanger, NodeTourist,
	parseLogicalPath2astModule, parsePathFilename2astModule, Then)
from astToolkit.containers import (
	astModuleToIngredientsFunction, IngredientsFunction, IngredientsModule, LedgerOfImports)
from astToolkit.transformationTools import inlineFunctionDef, removeUnusedParameters, write_astModule
from hunterMakesPy import importLogicalPath2Identifier, raiseIfNone
from mapFolding import packageSettings
from mapFolding.someAssemblyRequired import default, DeReConstructField2ast, IfThis, ShatteredDataclass
from mapFolding.someAssemblyRequired.makingModules_count import (
	makeMapFoldingNumba, makeTheorem2, numbaOnTheorem2, trimTheorem2)
from mapFolding.someAssemblyRequired.makingModules_doTheNeedful import makeInitializeState
from mapFolding.someAssemblyRequired.toolkitMakeModules import getModule, getPathFilename
from mapFolding.someAssemblyRequired.toolkitNumba import decorateCallableWithNumba, parametersNumbaLight
from mapFolding.someAssemblyRequired.transformationTools import (
	removeDataclassFromFunction, shatter_dataclassesDOTdataclass, unpackDataclassCallFunctionRepackDataclass)
from pathlib import PurePath
from typing import Any, TYPE_CHECKING
import ast
import dataclasses

if TYPE_CHECKING:
	from collections.abc import Sequence

def makeDaoOfMapFoldingParallelNumba(astModule: ast.Module, identifierModule: str, identifierCallable: str | None = None, logicalPathInfix: identifierDotAttribute | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:  # noqa: ARG001
	"""Generate parallel implementation with concurrent execution and task division.

	Parameters
	----------
	astModule : ast.Module
		Source module containing the base algorithm.
	moduleIdentifier : str
		Name for the generated parallel module.
	callableIdentifier : str | None = None
		Name for the core parallel counting function.
	logicalPathInfix : PathLike[str] | PurePath | str | None = None
		Directory path for organizing the generated module.
	sourceCallableDispatcher : str | None = None
		Optional dispatcher function identifier.

	Returns
	-------
	pathFilename : PurePath
		Filesystem path where the parallel module was written.

	"""
	sourceCallableIdentifier = default['function']['counting']
	if identifierCallable is None:
		identifierCallable = sourceCallableIdentifier
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = identifierCallable

	dataclassName: ast.expr = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.annotation)).captureLastMatch(ingredientsFunction.astFunctionDef))
	dataclassIdentifier: str = raiseIfNone(NodeTourist(Be.Name, Then.extractIt(DOT.id)).captureLastMatch(dataclassName))

	dataclassLogicalPathModule = None
	for moduleWithLogicalPath, listNameTuples in ingredientsFunction.imports._dictionaryImportFrom.items():  # noqa: SLF001
		for nameTuple in listNameTuples:
			if nameTuple[0] == dataclassIdentifier:
				dataclassLogicalPathModule = moduleWithLogicalPath
				break
		if dataclassLogicalPathModule:
			break
	if dataclassLogicalPathModule is None:
		raise Exception  # noqa: TRY002
	dataclassInstanceIdentifier: identifierDotAttribute = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))
	shatteredDataclass: ShatteredDataclass = shatter_dataclassesDOTdataclass(dataclassLogicalPathModule, dataclassIdentifier, dataclassInstanceIdentifier)

# START add the parallel state fields to the count function ------------------------------------------------
	dataclassBaseFields: tuple[dataclasses.Field[Any], ...] = dataclasses.fields(importLogicalPath2Identifier(dataclassLogicalPathModule, dataclassIdentifier))
	dataclassIdentifierParallel: identifierDotAttribute = 'Parallel' + dataclassIdentifier
	dataclassFieldsParallel: tuple[dataclasses.Field[Any], ...] = dataclasses.fields(importLogicalPath2Identifier(dataclassLogicalPathModule, dataclassIdentifierParallel))
	onlyParallelFields: list[dataclasses.Field[Any]] = [field for field in dataclassFieldsParallel if field.name not in [fieldBase.name for fieldBase in dataclassBaseFields]]

	Official_fieldOrder: list[str] = []
	dictionaryDeReConstruction: dict[str, DeReConstructField2ast] = {}

	dataclassClassDef: ast.ClassDef | None = extractClassDef(parseLogicalPath2astModule(dataclassLogicalPathModule), dataclassIdentifierParallel)
	if not dataclassClassDef:
		message = f"I could not find `{dataclassIdentifierParallel = }` in `{dataclassLogicalPathModule = }`."
		raise ValueError(message)

	for aField in onlyParallelFields:
		Official_fieldOrder.append(aField.name)
		dictionaryDeReConstruction[aField.name] = DeReConstructField2ast(dataclassLogicalPathModule, dataclassClassDef, dataclassInstanceIdentifier, aField)

	shatteredDataclassParallel = ShatteredDataclass(
		countingVariableAnnotation=shatteredDataclass.countingVariableAnnotation,
		countingVariableName=shatteredDataclass.countingVariableName,
		field2AnnAssign={**shatteredDataclass.field2AnnAssign, **{dictionaryDeReConstruction[field].name: dictionaryDeReConstruction[field].astAnnAssignConstructor for field in Official_fieldOrder}},
		Z0Z_field2AnnAssign={**shatteredDataclass.Z0Z_field2AnnAssign, **{dictionaryDeReConstruction[field].name: dictionaryDeReConstruction[field].Z0Z_hack for field in Official_fieldOrder}},
		list_argAnnotated4ArgumentsSpecification=shatteredDataclass.list_argAnnotated4ArgumentsSpecification + [dictionaryDeReConstruction[field].ast_argAnnotated for field in Official_fieldOrder],
		list_keyword_field__field4init=shatteredDataclass.list_keyword_field__field4init + [dictionaryDeReConstruction[field].ast_keyword_field__field for field in Official_fieldOrder if dictionaryDeReConstruction[field].init],
		listAnnotations=shatteredDataclass.listAnnotations + [dictionaryDeReConstruction[field].astAnnotation for field in Official_fieldOrder],
		listName4Parameters=shatteredDataclass.listName4Parameters + [dictionaryDeReConstruction[field].astName for field in Official_fieldOrder],
		listUnpack=shatteredDataclass.listUnpack + [Make.AnnAssign(dictionaryDeReConstruction[field].astName, dictionaryDeReConstruction[field].astAnnotation, dictionaryDeReConstruction[field].ast_nameDOTname) for field in Official_fieldOrder],
		map_stateDOTfield2Name={**shatteredDataclass.map_stateDOTfield2Name, **{dictionaryDeReConstruction[field].ast_nameDOTname: dictionaryDeReConstruction[field].astName for field in Official_fieldOrder}},
		)
	shatteredDataclassParallel.fragments4AssignmentOrParameters = Make.Tuple(shatteredDataclassParallel.listName4Parameters, Make.Store())
	shatteredDataclassParallel.repack = Make.Assign([Make.Name(dataclassInstanceIdentifier)], value=Make.Call(Make.Name(dataclassIdentifierParallel), list_keyword=shatteredDataclassParallel.list_keyword_field__field4init))
	shatteredDataclassParallel.signatureReturnAnnotation = Make.Subscript(Make.Name('tuple'), Make.Tuple(shatteredDataclassParallel.listAnnotations))

	shatteredDataclassParallel.imports.update(*(dictionaryDeReConstruction[field].ledger for field in Official_fieldOrder))
	shatteredDataclassParallel.imports.addImportFrom_asStr(dataclassLogicalPathModule, dataclassIdentifierParallel)
	shatteredDataclassParallel.imports.update(shatteredDataclass.imports)
	shatteredDataclassParallel.imports.removeImportFrom(dataclassLogicalPathModule, dataclassIdentifier)

# END add the parallel state fields to the count function ------------------------------------------------

	ingredientsFunction.imports.update(shatteredDataclassParallel.imports)
	ingredientsFunction: IngredientsFunction = removeDataclassFromFunction(ingredientsFunction, shatteredDataclassParallel)

# START add the parallel logic to the count function ------------------------------------------------

	findThis = Be.While.testIs(Be.Compare.leftIs(IfThis.isNameIdentifier('leafConnectee')))
	captureCountGapsCodeBlock: NodeTourist[ast.While, Sequence[ast.stmt]] = NodeTourist(findThis, doThat = Then.extractIt(DOT.body))
	countGapsCodeBlock: Sequence[ast.stmt] = raiseIfNone(captureCountGapsCodeBlock.captureLastMatch(ingredientsFunction.astFunctionDef))

	thisIsMyTaskIndexCodeBlock = ast.If(ast.BoolOp(ast.Or()
		, values=[ast.Compare(ast.Name('leaf1ndex'), ops=[ast.NotEq()], comparators=[ast.Name('taskDivisions')])
				, ast.Compare(Make.Mod.join([ast.Name('leafConnectee'), ast.Name('taskDivisions')]), ops=[ast.Eq()], comparators=[ast.Name('taskIndex')])])
	, body=list(countGapsCodeBlock[0:-1]))

	countGapsCodeBlockNew: list[ast.stmt] = [thisIsMyTaskIndexCodeBlock, countGapsCodeBlock[-1]]
	NodeChanger[ast.While, hasDOTbody](findThis, doThat = Grab.bodyAttribute(Then.replaceWith(countGapsCodeBlockNew))).visit(ingredientsFunction.astFunctionDef)

# END add the parallel logic to the count function ------------------------------------------------

	ingredientsFunction.removeUnusedParameters()

	ingredientsFunction = decorateCallableWithNumba(ingredientsFunction, parametersNumbaLight)

# START unpack/repack the dataclass function ------------------------------------------------
	sourceCallableIdentifier = default['function']['dispatcher']

	unRepackDataclass: IngredientsFunction = astModuleToIngredientsFunction(astModule, sourceCallableIdentifier)
	unRepackDataclass.astFunctionDef.name = 'unRepack' + dataclassIdentifierParallel
	unRepackDataclass.imports.update(shatteredDataclassParallel.imports)
	NodeChanger(
			findThis = Be.arg.annotationIs(Be.Name.idIs(lambda thisAttribute: thisAttribute == dataclassIdentifier))
			, doThat = Grab.annotationAttribute(Grab.idAttribute(Then.replaceWith(dataclassIdentifierParallel)))
		).visit(unRepackDataclass.astFunctionDef)
	unRepackDataclass.astFunctionDef.returns = Make.Name(dataclassIdentifierParallel)
	targetCallableIdentifier: identifierDotAttribute = ingredientsFunction.astFunctionDef.name
	unRepackDataclass = unpackDataclassCallFunctionRepackDataclass(unRepackDataclass, targetCallableIdentifier, shatteredDataclassParallel)

	astTuple: ast.Tuple = raiseIfNone(NodeTourist[ast.Return, ast.Tuple](Be.Return, Then.extractIt(DOT.value)).captureLastMatch(ingredientsFunction.astFunctionDef)) # pyright: ignore[reportArgumentType]
	astTuple.ctx = Make.Store()
	changeAssignCallToTarget: NodeChanger[ast.Assign, ast.Assign] = NodeChanger(
		findThis = Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier))
		, doThat = Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(targetCallableIdentifier), astTuple.elts)))
	)
	changeAssignCallToTarget.visit(unRepackDataclass.astFunctionDef)

	ingredientsDoTheNeedful: IngredientsFunction = IngredientsFunction(
		astFunctionDef = Make.FunctionDef('doTheNeedful'
			, argumentSpecification=Make.arguments(list_arg=[Make.arg('state', annotation=Make.Name(dataclassIdentifierParallel)), Make.arg('concurrencyLimit', annotation=Make.Name('int'))])
			, body=[Make.Assign([Make.Name('stateParallel', Make.Store())], value=Make.Call(Make.Name('deepcopy'), listParameters=[Make.Name('state')]))
				, Make.AnnAssign(Make.Name('listStatesParallel', Make.Store()), annotation=Make.Subscript(value=Make.Name('list'), slice=Make.Name(dataclassIdentifierParallel))
					, value=Make.Mult.join([Make.List([Make.Name('stateParallel')]), Make.Attribute(Make.Name('stateParallel'), 'taskDivisions')]))
				, Make.AnnAssign(Make.Name('groupsOfFoldsTotal', Make.Store()), annotation=Make.Name('int'), value=Make.Constant(value=0))

				, Make.AnnAssign(Make.Name('dictionaryConcurrency', Make.Store()), annotation=Make.Subscript(value=Make.Name('dict'), slice=Make.Tuple([Make.Name('int'), Make.Subscript(value=Make.Name('ConcurrentFuture'), slice=Make.Name(dataclassIdentifierParallel))])), value=Make.Dict())
				, Make.With(items=[Make.withitem(context_expr=Make.Call(Make.Name('ProcessPoolExecutor'), listParameters=[Make.Name('concurrencyLimit')]), optional_vars=Make.Name('concurrencyManager', Make.Store()))]
					, body=[Make.For(Make.Name('indexSherpa', Make.Store()), iter=Make.Call(Make.Name('range'), listParameters=[Make.Attribute(Make.Name('stateParallel'), 'taskDivisions')])
							, body=[Make.Assign([Make.Name('state', Make.Store())], value=Make.Call(Make.Name('deepcopy'), listParameters=[Make.Name('stateParallel')]))
								, Make.Assign([Make.Attribute(Make.Name('state'), 'taskIndex', context=Make.Store())], value=Make.Name('indexSherpa'))
								, Make.Assign([Make.Subscript(Make.Name('dictionaryConcurrency'), slice=Make.Name('indexSherpa'), context=Make.Store())], value=Make.Call(Make.Attribute(Make.Name('concurrencyManager'), 'submit'), listParameters=[Make.Name(unRepackDataclass.astFunctionDef.name), Make.Name('state')]))])
						, Make.For(Make.Name('indexSherpa', Make.Store()), iter=Make.Call(Make.Name('range'), listParameters=[Make.Attribute(Make.Name('stateParallel'), 'taskDivisions')])
							, body=[Make.Assign([Make.Subscript(Make.Name('listStatesParallel'), slice=Make.Name('indexSherpa'), context=Make.Store())], value=Make.Call(Make.Attribute(Make.Subscript(Make.Name('dictionaryConcurrency'), slice=Make.Name('indexSherpa')), 'result')))
								, Make.AugAssign(Make.Name('groupsOfFoldsTotal', Make.Store()), op=ast.Add(), value=Make.Attribute(Make.Subscript(Make.Name('listStatesParallel'), slice=Make.Name('indexSherpa')), 'groupsOfFolds'))])])

				, Make.AnnAssign(Make.Name('foldsTotal', Make.Store()), annotation=Make.Name('int'), value=Make.Mult.join([Make.Name('groupsOfFoldsTotal'), Make.Attribute(Make.Name('stateParallel'), 'leavesTotal')]))
				, Make.Return(Make.Tuple([Make.Name('foldsTotal'), Make.Name('listStatesParallel')]))]
			, returns=Make.Subscript(Make.Name('tuple'), slice=Make.Tuple([Make.Name('int'), Make.Subscript(Make.Name('list'), slice=Make.Name(dataclassIdentifierParallel))])))
		, imports = LedgerOfImports(Make.Module([Make.ImportFrom('concurrent.futures', list_alias=[Make.alias('Future', asName='ConcurrentFuture'), Make.alias('ProcessPoolExecutor')]),
			Make.ImportFrom('copy', list_alias=[Make.alias('deepcopy')]),
			Make.ImportFrom('multiprocessing', list_alias=[Make.alias('set_start_method', asName='multiprocessing_set_start_method')])])
		)
	)

	ingredientsModule = IngredientsModule([ingredientsFunction, unRepackDataclass, ingredientsDoTheNeedful]
						, prologue = Make.Module([Make.If(test=Make.Compare(left=Make.Name('__name__'), ops=[Make.Eq()], comparators=[Make.Constant('__main__')]), body=[Make.Expr(Make.Call(Make.Name('multiprocessing_set_start_method'), listParameters=[Make.Constant('spawn')]))])])
	)
	ingredientsModule.removeImportFromModule('numpy')

	pathFilename: PurePath = getPathFilename(packageSettings.pathPackage, logicalPathInfix, identifierModule)

	ingredientsModule.write_astModule(pathFilename, packageSettings.identifierPackage)

	return pathFilename

def makeMapFoldingModules() -> None:
	"""Make multidimensional map folding modules."""
	astModule = getModule(logicalPathInfix='algorithms')
	pathFilename: PurePath = makeMapFoldingNumba(astModule, 'daoOfMapFoldingNumba', None, default['logicalPath']['synthetic'], default['function']['dispatcher'])

	astModule = getModule(logicalPathInfix='algorithms')
	pathFilename = makeDaoOfMapFoldingParallelNumba(astModule, 'countParallelNumba', None, default['logicalPath']['synthetic'], default['function']['dispatcher'])

	astModule: ast.Module = getModule(logicalPathInfix='algorithms')
	makeInitializeState(astModule, default['module']['initializeState'], default['function']['initializeState'], default['logicalPath']['synthetic'])

	astModule = getModule(logicalPathInfix='algorithms')
	pathFilename = makeTheorem2(astModule, 'theorem2', None, default['logicalPath']['synthetic'], default['function']['dispatcher'])

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = trimTheorem2(astModule, 'theorem2Trimmed', None, default['logicalPath']['synthetic'], default['function']['dispatcher'])

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = numbaOnTheorem2(astModule, 'theorem2Numba', None, default['logicalPath']['synthetic'], default['function']['dispatcher'])

if __name__ == '__main__':
	makeMapFoldingModules()
