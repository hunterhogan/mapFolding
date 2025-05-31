from astToolkit import (
	astModuleToIngredientsFunction, Be, ClassIsAndAttribute, DOT, extractClassDef, extractFunctionDef,
	Grab, identifierDotAttribute, IngredientsFunction, IngredientsModule, LedgerOfImports, Make,
	NodeChanger, NodeTourist, parseLogicalPath2astModule, parsePathFilename2astModule, Then,
)
from astToolkit.transformationTools import (
	inlineFunctionDef, removeUnusedParameters, write_astModule,
)
from collections.abc import Sequence
from mapFolding import packageSettings
from mapFolding.someAssemblyRequired import (
	DeReConstructField2ast, IfThis, ShatteredDataclass, sourceCallableDispatcherDEFAULT,
)
from mapFolding.someAssemblyRequired.infoBooth import (
	algorithmSourceModuleDEFAULT, dataPackingModuleIdentifierDEFAULT, logicalPathInfixDEFAULT,
	sourceCallableIdentifierDEFAULT, theCountingIdentifierDEFAULT,
)
from mapFolding.someAssemblyRequired.toolkitNumba import (
	decorateCallableWithNumba, parametersNumbaLight,
)
from mapFolding.someAssemblyRequired.transformationTools import (
	removeDataclassFromFunction, shatter_dataclassesDOTdataclass,
	unpackDataclassCallFunctionRepackDataclass,
)
from os import PathLike
from pathlib import PurePath
from typing import cast
from Z0Z_tools import importLogicalPath2Callable, raiseIfNone
import ast
import dataclasses

def findDataclass(ingredientsFunction: IngredientsFunction) -> tuple[str, str, str]:
	dataclassName: ast.expr = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.annotation)).captureLastMatch(ingredientsFunction.astFunctionDef))
	dataclassIdentifier: str = raiseIfNone(NodeTourist(Be.Name, Then.extractIt(DOT.id)).captureLastMatch(dataclassName))
	dataclassLogicalPathModule = None
	for moduleWithLogicalPath, listNameTuples in ingredientsFunction.imports._dictionaryImportFrom.items():
		for nameTuple in listNameTuples:
			if nameTuple[0] == dataclassIdentifier:
				dataclassLogicalPathModule = moduleWithLogicalPath
				break
		if dataclassLogicalPathModule:
			break
	dataclassInstanceIdentifier = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))
	return raiseIfNone(dataclassLogicalPathModule), dataclassIdentifier, dataclassInstanceIdentifier

def _getLogicalPath(packageName: str | None = None, logicalPathInfix: str | None = None, moduleIdentifier: str | None = None, *modules: str) -> identifierDotAttribute:
	listLogicalPathParts: list[str] = []
	if packageName:
		listLogicalPathParts.append(packageName)
	if logicalPathInfix:
		listLogicalPathParts.append(logicalPathInfix)
	if moduleIdentifier:
		listLogicalPathParts.append(moduleIdentifier)
	if modules:
		listLogicalPathParts.extend(modules)
	logicalPath = '.'.join(listLogicalPathParts)
	return logicalPath

def getModule(packageName: str | None = packageSettings.packageName, logicalPathInfix: str | None = logicalPathInfixDEFAULT, moduleIdentifier: str | None = algorithmSourceModuleDEFAULT) -> ast.Module:
	logicalPathSourceModule = _getLogicalPath(packageName, logicalPathInfix, moduleIdentifier)
	astModule = parseLogicalPath2astModule(logicalPathSourceModule)
	return astModule

def _getPathFilename(pathRoot: PathLike[str] | PurePath | None = packageSettings.pathPackage, logicalPathInfix: PathLike[str] | PurePath | str | None = None, moduleIdentifier: str = '', fileExtension: str = packageSettings.fileExtension) -> PurePath:
	pathFilename = PurePath(moduleIdentifier + fileExtension)
	if logicalPathInfix:
		pathFilename = PurePath(logicalPathInfix, pathFilename)
	if pathRoot:
		pathFilename = PurePath(pathRoot, pathFilename)
	return pathFilename

def makeInitializeGroupsOfFolds(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:
	sourceCallableIdentifier = sourceCallableIdentifierDEFAULT
	if callableIdentifier is None:
		callableIdentifier = sourceCallableIdentifier
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = callableIdentifier

	dataclassInstanceIdentifier = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))
	theCountingIdentifier = theCountingIdentifierDEFAULT

	findThis = IfThis.isWhileAttributeNamespaceIdentifierGreaterThan0(dataclassInstanceIdentifier, 'leaf1ndex')
	doThat = Grab.testAttribute(Grab.andDoAllOf([Grab.opsAttribute(Then.replaceWith([ast.Eq()])), Grab.leftAttribute(Grab.attrAttribute(Then.replaceWith(theCountingIdentifier)))])) # pyright: ignore[reportArgumentType]
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef.body[0])

	ingredientsModule = IngredientsModule(ingredientsFunction)
	pathFilename = _getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)
	write_astModule(ingredientsModule, pathFilename, packageSettings.packageName)

	return pathFilename

def makeDaoOfMapFolding(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:
	sourceCallableIdentifier = sourceCallableIdentifierDEFAULT
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))

	if callableIdentifier is None:
		callableIdentifier = sourceCallableIdentifier
	ingredientsFunction.astFunctionDef.name = callableIdentifier

	shatteredDataclass = shatter_dataclassesDOTdataclass(*findDataclass(ingredientsFunction))

	ingredientsFunction.imports.update(shatteredDataclass.imports)
	ingredientsFunction = removeDataclassFromFunction(ingredientsFunction, shatteredDataclass)
	ingredientsFunction = removeUnusedParameters(ingredientsFunction)
	ingredientsFunction = decorateCallableWithNumba(ingredientsFunction, parametersNumbaLight)

	ingredientsModule = IngredientsModule(ingredientsFunction)

	if sourceCallableDispatcher is not None:

		ingredientsFunctionDispatcher: IngredientsFunction = astModuleToIngredientsFunction(astModule, sourceCallableDispatcher)
		ingredientsFunctionDispatcher.imports.update(shatteredDataclass.imports)
		targetCallableIdentifier = ingredientsFunction.astFunctionDef.name
		ingredientsFunctionDispatcher = unpackDataclassCallFunctionRepackDataclass(ingredientsFunctionDispatcher, targetCallableIdentifier, shatteredDataclass)
		astTuple = raiseIfNone(NodeTourist(Be.Return, Then.extractIt(DOT.value)).captureLastMatch(ingredientsFunction.astFunctionDef))
		cast(ast.Tuple, astTuple).ctx = ast.Store()

		findThis = ClassIsAndAttribute.valueIs(ast.Assign, IfThis.isCallIdentifier(targetCallableIdentifier))
		doThat = Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(targetCallableIdentifier), cast(ast.Tuple, astTuple).elts)))
		changeAssignCallToTarget = NodeChanger(findThis, doThat)
		changeAssignCallToTarget.visit(ingredientsFunctionDispatcher.astFunctionDef)

		ingredientsModule.appendIngredientsFunction(ingredientsFunctionDispatcher)

	ingredientsModule.removeImportFromModule('numpy')

	pathFilename = _getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

	write_astModule(ingredientsModule, pathFilename, packageSettings.packageName)

	return pathFilename

def makeDaoOfMapFoldingParallel(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:
	sourceCallableIdentifier = sourceCallableIdentifierDEFAULT
	if callableIdentifier is None:
		callableIdentifier = sourceCallableIdentifier
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = callableIdentifier

	dataclassName: ast.expr = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.annotation)).captureLastMatch(ingredientsFunction.astFunctionDef))
	dataclassIdentifier: str = raiseIfNone(NodeTourist(Be.Name, Then.extractIt(DOT.id)).captureLastMatch(dataclassName))

	dataclassLogicalPathModule = None
	for moduleWithLogicalPath, listNameTuples in ingredientsFunction.imports._dictionaryImportFrom.items():
		for nameTuple in listNameTuples:
			if nameTuple[0] == dataclassIdentifier:
				dataclassLogicalPathModule = moduleWithLogicalPath
				break
		if dataclassLogicalPathModule:
			break
	if dataclassLogicalPathModule is None:
		raise Exception
	dataclassInstanceIdentifier = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))
	shatteredDataclass = shatter_dataclassesDOTdataclass(dataclassLogicalPathModule, dataclassIdentifier, dataclassInstanceIdentifier)

	# Start add the parallel state fields to the count function ================================================
	dataclassBaseFields = dataclasses.fields(importLogicalPath2Callable(dataclassLogicalPathModule, dataclassIdentifier))  # pyright: ignore [reportArgumentType]
	dataclassIdentifierParallel = 'Parallel' + dataclassIdentifier
	dataclassFieldsParallel = dataclasses.fields(importLogicalPath2Callable(dataclassLogicalPathModule, dataclassIdentifierParallel))  # pyright: ignore [reportArgumentType]
	onlyParallelFields = [field for field in dataclassFieldsParallel if field.name not in [fieldBase.name for fieldBase in dataclassBaseFields]]

	Official_fieldOrder: list[str] = []
	dictionaryDeReConstruction: dict[str, DeReConstructField2ast] = {}

	dataclassClassDef = extractClassDef(parseLogicalPath2astModule(dataclassLogicalPathModule), dataclassIdentifierParallel)
	if not isinstance(dataclassClassDef, ast.ClassDef):
		raise ValueError(f"I could not find `{dataclassIdentifierParallel = }` in `{dataclassLogicalPathModule = }`.")

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
	shatteredDataclassParallel.fragments4AssignmentOrParameters = Make.Tuple(shatteredDataclassParallel.listName4Parameters, ast.Store())
	shatteredDataclassParallel.repack = Make.Assign([Make.Name(dataclassInstanceIdentifier)], value=Make.Call(Make.Name(dataclassIdentifierParallel), list_keyword=shatteredDataclassParallel.list_keyword_field__field4init))
	shatteredDataclassParallel.signatureReturnAnnotation = Make.Subscript(Make.Name('tuple'), Make.Tuple(shatteredDataclassParallel.listAnnotations))

	shatteredDataclassParallel.imports.update(*(dictionaryDeReConstruction[field].ledger for field in Official_fieldOrder))
	shatteredDataclassParallel.imports.addImportFrom_asStr(dataclassLogicalPathModule, dataclassIdentifierParallel)
	shatteredDataclassParallel.imports.update(shatteredDataclass.imports)
	shatteredDataclassParallel.imports.removeImportFrom(dataclassLogicalPathModule, dataclassIdentifier)

	# End add the parallel state fields to the count function ================================================

	ingredientsFunction.imports.update(shatteredDataclassParallel.imports)
	ingredientsFunction = removeDataclassFromFunction(ingredientsFunction, shatteredDataclassParallel)

	# Start add the parallel logic to the count function ================================================

	findThis = ClassIsAndAttribute.testIs(ast.While, ClassIsAndAttribute.leftIs(ast.Compare, IfThis.isNameIdentifier('leafConnectee')))
	doThat = Then.extractIt(DOT.body)
	captureCountGapsCodeBlock: NodeTourist[ast.While, Sequence[ast.stmt]] = NodeTourist(findThis, doThat)
	countGapsCodeBlock = raiseIfNone(captureCountGapsCodeBlock.captureLastMatch(ingredientsFunction.astFunctionDef))

	thisIsMyTaskIndexCodeBlock = ast.If(ast.BoolOp(ast.Or()
		, values=[ast.Compare(ast.Name('leaf1ndex'), ops=[ast.NotEq()], comparators=[ast.Name('taskDivisions')])
				, ast.Compare(Make.Mod.join([ast.Name('leafConnectee'), ast.Name('taskDivisions')]), ops=[ast.Eq()], comparators=[ast.Name('taskIndex')])])
	, body=list(countGapsCodeBlock[0:-1]))

	countGapsCodeBlockNew: list[ast.stmt] = [thisIsMyTaskIndexCodeBlock, countGapsCodeBlock[-1]]

	doThat = Grab.bodyAttribute(Then.replaceWith(countGapsCodeBlockNew))
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef)

	# End add the parallel logic to the count function ================================================

	ingredientsFunction = removeUnusedParameters(ingredientsFunction)

	ingredientsFunction = decorateCallableWithNumba(ingredientsFunction, parametersNumbaLight)

	# Start unpack/repack the dataclass function ================================================
	sourceCallableIdentifier = sourceCallableDispatcherDEFAULT

	unRepackDataclass: IngredientsFunction = astModuleToIngredientsFunction(astModule, sourceCallableIdentifier)
	unRepackDataclass.astFunctionDef.name = 'unRepack' + dataclassIdentifierParallel
	unRepackDataclass.imports.update(shatteredDataclassParallel.imports)
	findThis = ClassIsAndAttribute.annotationIs(ast.arg, IfThis.isNameIdentifier(dataclassIdentifier))
	doThat = Grab.annotationAttribute(Grab.idAttribute(Then.replaceWith(dataclassIdentifierParallel))) # pyright: ignore[reportArgumentType]
	NodeChanger(findThis, doThat).visit(unRepackDataclass.astFunctionDef)
	unRepackDataclass.astFunctionDef.returns = Make.Name(dataclassIdentifierParallel)
	targetCallableIdentifier = ingredientsFunction.astFunctionDef.name
	unRepackDataclass = unpackDataclassCallFunctionRepackDataclass(unRepackDataclass, targetCallableIdentifier, shatteredDataclassParallel)

	astTuple = raiseIfNone(NodeTourist(Be.Return, Then.extractIt(DOT.value)).captureLastMatch(ingredientsFunction.astFunctionDef))
	cast(ast.Tuple, astTuple).ctx = ast.Store()
	findThis = ClassIsAndAttribute.valueIs(ast.Assign, IfThis.isCallIdentifier(targetCallableIdentifier))
	doThat = Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(targetCallableIdentifier), cast(ast.Tuple, astTuple).elts)))
	changeAssignCallToTarget = NodeChanger(findThis, doThat)
	changeAssignCallToTarget.visit(unRepackDataclass.astFunctionDef)

	ingredientsDoTheNeedful: IngredientsFunction = IngredientsFunction(
		astFunctionDef = Make.FunctionDef('doTheNeedful'
			, argumentSpecification=Make.arguments(list_arg=[Make.arg('state', annotation=Make.Name(dataclassIdentifierParallel)), Make.arg('concurrencyLimit', annotation=Make.Name('int'))])
			, body=[Make.Assign([Make.Name('stateParallel', ast.Store())], value=Make.Call(Make.Name('deepcopy'), listParameters=[Make.Name('state')]))
				, Make.AnnAssign(Make.Name('listStatesParallel', ast.Store()), annotation=Make.Subscript(value=Make.Name('list'), slice=Make.Name(dataclassIdentifierParallel))
					, value=Make.Mult.join([Make.List([Make.Name('stateParallel')]), Make.Attribute(Make.Name('stateParallel'), 'taskDivisions')]))
				, Make.AnnAssign(Make.Name('groupsOfFoldsTotal', ast.Store()), annotation=Make.Name('int'), value=Make.Constant(value=0))

				, Make.AnnAssign(Make.Name('dictionaryConcurrency', ast.Store()), annotation=Make.Subscript(value=Make.Name('dict'), slice=Make.Tuple([Make.Name('int'), Make.Subscript(value=Make.Name('ConcurrentFuture'), slice=Make.Name(dataclassIdentifierParallel))])), value=Make.Dict())
				, Make.With(items=[Make.withitem(context_expr=Make.Call(Make.Name('ProcessPoolExecutor'), listParameters=[Make.Name('concurrencyLimit')]), optional_vars=Make.Name('concurrencyManager', ast.Store()))]
					, body=[Make.For(Make.Name('indexSherpa', ast.Store()), iter=Make.Call(Make.Name('range'), listParameters=[Make.Attribute(Make.Name('stateParallel'), 'taskDivisions')])
							, body=[Make.Assign([Make.Name('state', ast.Store())], value=Make.Call(Make.Name('deepcopy'), listParameters=[Make.Name('stateParallel')]))
								, Make.Assign([Make.Attribute(Make.Name('state'), 'taskIndex', context=ast.Store())], value=Make.Name('indexSherpa'))
								, Make.Assign([Make.Subscript(Make.Name('dictionaryConcurrency'), slice=Make.Name('indexSherpa'), context=ast.Store())], value=Make.Call(Make.Attribute(Make.Name('concurrencyManager'), 'submit'), listParameters=[Make.Name(unRepackDataclass.astFunctionDef.name), Make.Name('state')]))])
						, Make.For(Make.Name('indexSherpa', ast.Store()), iter=Make.Call(Make.Name('range'), listParameters=[Make.Attribute(Make.Name('stateParallel'), 'taskDivisions')])
							, body=[Make.Assign([Make.Subscript(Make.Name('listStatesParallel'), slice=Make.Name('indexSherpa'), context=ast.Store())], value=Make.Call(Make.Attribute(Make.Subscript(Make.Name('dictionaryConcurrency'), slice=Make.Name('indexSherpa')), 'result')))
								, Make.AugAssign(Make.Name('groupsOfFoldsTotal', ast.Store()), op=ast.Add(), value=Make.Attribute(Make.Subscript(Make.Name('listStatesParallel'), slice=Make.Name('indexSherpa')), 'groupsOfFolds'))])])

				, Make.AnnAssign(Make.Name('foldsTotal', ast.Store()), annotation=Make.Name('int'), value=Make.Mult.join([Make.Name('groupsOfFoldsTotal'), Make.Attribute(Make.Name('stateParallel'), 'leavesTotal')]))
				, Make.Return(Make.Tuple([Make.Name('foldsTotal'), Make.Name('listStatesParallel')]))]
			, returns=Make.Subscript(Make.Name('tuple'), slice=Make.Tuple([Make.Name('int'), Make.Subscript(Make.Name('list'), slice=Make.Name(dataclassIdentifierParallel))])))
		, imports = LedgerOfImports(Make.Module([Make.ImportFrom('concurrent.futures', list_alias=[Make.alias('Future', asName='ConcurrentFuture'), Make.alias('ProcessPoolExecutor')]),
			Make.ImportFrom('copy', list_alias=[Make.alias('deepcopy')]),
			Make.ImportFrom('multiprocessing', list_alias=[Make.alias('set_start_method', asName='multiprocessing_set_start_method')]),])
		)
	)

	ingredientsModule = IngredientsModule([ingredientsFunction, unRepackDataclass, ingredientsDoTheNeedful]
						, prologue = Make.Module([Make.If(test=Make.Compare(left=Make.Name('__name__'), ops=[Make.Eq()], comparators=[Make.Constant('__main__')]), body=[Make.Expr(Make.Call(Make.Name('multiprocessing_set_start_method'), listParameters=[Make.Constant('spawn')]))])])
	)
	ingredientsModule.removeImportFromModule('numpy')

	pathFilename = _getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

	write_astModule(ingredientsModule, pathFilename, packageSettings.packageName)
	return pathFilename

def makeTheorem2(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:
	sourceCallableIdentifier = sourceCallableIdentifierDEFAULT
	if callableIdentifier is None:
		callableIdentifier = sourceCallableIdentifier
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = callableIdentifier

	dataclassInstanceIdentifier = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))

	findThis = IfThis.isWhileAttributeNamespaceIdentifierGreaterThan0(dataclassInstanceIdentifier, 'leaf1ndex')
	doThat = Grab.testAttribute(Grab.comparatorsAttribute(Then.replaceWith([Make.Constant(4)]))) # pyright: ignore[reportArgumentType]
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef)

	findThis = IfThis.isIfAttributeNamespaceIdentifierGreaterThan0(dataclassInstanceIdentifier, 'leaf1ndex')
	doThat = Then.extractIt(DOT.body)
	insertLeaf = NodeTourist(findThis, doThat).captureLastMatch(ingredientsFunction.astFunctionDef)
	findThis = IfThis.isIfAttributeNamespaceIdentifierGreaterThan0(dataclassInstanceIdentifier, 'leaf1ndex')
	doThat = Then.replaceWith(insertLeaf)
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef)

	findThis = IfThis.isAttributeNamespaceIdentifierGreaterThan0(dataclassInstanceIdentifier, 'leaf1ndex')
	doThat = Then.removeIt
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef)

	findThis = IfThis.isAttributeNamespaceIdentifierLessThanOrEqual0(dataclassInstanceIdentifier, 'leaf1ndex')
	doThat = Then.removeIt
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef)

	theCountingIdentifier = theCountingIdentifierDEFAULT
	doubleTheCount = Make.AugAssign(Make.Attribute(ast.Name(dataclassInstanceIdentifier), theCountingIdentifier), ast.Mult(), Make.Constant(2))
	findThis = Be.Return
	doThat = Then.insertThisAbove([doubleTheCount])
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef)

	ingredientsModule = IngredientsModule(ingredientsFunction)

	if sourceCallableDispatcher is not None:
		raise NotImplementedError('sourceCallableDispatcher is not implemented yet')

	pathFilename = _getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

	write_astModule(ingredientsModule, pathFilename, packageSettings.packageName)

	return pathFilename

def trimTheorem2(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:
	sourceCallableIdentifier = sourceCallableIdentifierDEFAULT
	if callableIdentifier is None:
		callableIdentifier = sourceCallableIdentifier
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = callableIdentifier

	dataclassInstanceIdentifier = raiseIfNone(NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))

	findThis = IfThis.isIfUnaryNotAttributeNamespaceIdentifier(dataclassInstanceIdentifier, 'dimensionsUnconstrained')
	doThat = Then.removeIt
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef)

	ingredientsModule = IngredientsModule(ingredientsFunction)
	ingredientsModule.removeImportFromModule('numpy')

	pathFilename = _getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

	write_astModule(ingredientsModule, pathFilename, packageSettings.packageName)

	return pathFilename

def numbaOnTheorem2(astModule: ast.Module, moduleIdentifier: str, callableIdentifier: str | None = None, logicalPathInfix: PathLike[str] | PurePath | str | None = None, sourceCallableDispatcher: str | None = None) -> PurePath:
	sourceCallableIdentifier = sourceCallableIdentifierDEFAULT
	if callableIdentifier is None:
		callableIdentifier = sourceCallableIdentifier
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))
	ingredientsFunction.astFunctionDef.name = callableIdentifier

	shatteredDataclass = shatter_dataclassesDOTdataclass(*findDataclass(ingredientsFunction))

	ingredientsFunction.imports.update(shatteredDataclass.imports)
	ingredientsFunction = removeDataclassFromFunction(ingredientsFunction, shatteredDataclass)
	ingredientsFunction = removeUnusedParameters(ingredientsFunction)
	ingredientsFunction = decorateCallableWithNumba(ingredientsFunction, parametersNumbaLight)

	ingredientsModule = IngredientsModule(ingredientsFunction)
	ingredientsModule.removeImportFromModule('numpy')

	pathFilename = _getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

	write_astModule(ingredientsModule, pathFilename, packageSettings.packageName)

	return pathFilename

def makeUnRePackDataclass(astImportFrom: ast.ImportFrom) -> None:
	callableIdentifierHARDCODED: str = 'sequential'

	algorithmSourceModule = algorithmSourceModuleDEFAULT
	sourceCallableIdentifier = sourceCallableDispatcherDEFAULT
	logicalPathSourceModule = '.'.join([packageSettings.packageName, algorithmSourceModule])

	logicalPathInfix = logicalPathInfixDEFAULT
	moduleIdentifier = dataPackingModuleIdentifierDEFAULT
	callableIdentifier = callableIdentifierHARDCODED

	ingredientsFunction: IngredientsFunction = astModuleToIngredientsFunction(parseLogicalPath2astModule(logicalPathSourceModule), sourceCallableIdentifier)
	ingredientsFunction.astFunctionDef.name = callableIdentifier

	shatteredDataclass = shatter_dataclassesDOTdataclass(*findDataclass(ingredientsFunction))

	ingredientsFunction.imports.update(shatteredDataclass.imports)
	ingredientsFunction.imports.addAst(astImportFrom)
	targetCallableIdentifier = astImportFrom.names[0].name
	ingredientsFunction = raiseIfNone(unpackDataclassCallFunctionRepackDataclass(ingredientsFunction, targetCallableIdentifier, shatteredDataclass))
	targetFunctionDef = raiseIfNone(extractFunctionDef(parseLogicalPath2astModule(raiseIfNone(astImportFrom.module)), targetCallableIdentifier))
	astTuple = raiseIfNone(NodeTourist(Be.Return, Then.extractIt(DOT.value)).captureLastMatch(targetFunctionDef))
	cast(ast.Tuple, astTuple).ctx = ast.Store()

	findThis = ClassIsAndAttribute.valueIs(ast.Assign, IfThis.isCallIdentifier(targetCallableIdentifier))
	doThat = Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(targetCallableIdentifier), cast(ast.Tuple, astTuple).elts)))
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef)

	ingredientsModule = IngredientsModule(ingredientsFunction)
	ingredientsModule.removeImportFromModule('numpy')

	pathFilename = _getPathFilename(packageSettings.pathPackage, logicalPathInfix, moduleIdentifier)

	write_astModule(ingredientsModule, pathFilename, packageSettings.packageName)

if __name__ == '__main__':
	astModule = getModule(logicalPathInfix=None)
	makeInitializeGroupsOfFolds(astModule, 'initializeCount', 'initializeGroupsOfFolds', logicalPathInfixDEFAULT)

	astModule = getModule(logicalPathInfix=None)
	pathFilename = makeDaoOfMapFolding(astModule, 'daoOfMapFolding', None, logicalPathInfixDEFAULT, sourceCallableDispatcherDEFAULT)

	astModule = getModule(logicalPathInfix=None)
	pathFilename = makeDaoOfMapFoldingParallel(astModule, 'countParallel', None, logicalPathInfixDEFAULT, sourceCallableDispatcherDEFAULT)

	astModule = getModule(logicalPathInfix=None)
	pathFilename = makeTheorem2(astModule, 'theorem2', None, logicalPathInfixDEFAULT, None)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = trimTheorem2(astModule, 'theorem2Trimmed', None, logicalPathInfixDEFAULT, None)

	astModule = parsePathFilename2astModule(pathFilename)
	pathFilename = numbaOnTheorem2(astModule, 'theorem2Numba', None, logicalPathInfixDEFAULT, None)

	astImportFrom = Make.ImportFrom(_getLogicalPath(packageSettings.packageName, logicalPathInfixDEFAULT, 'theorem2Numba'), list_alias=[Make.alias(sourceCallableIdentifierDEFAULT)])
	makeUnRePackDataclass(astImportFrom)
