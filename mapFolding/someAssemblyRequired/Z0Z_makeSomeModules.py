from collections.abc import Sequence
from astToolkit import ClassIsAndAttribute, extractClassDef
from mapFolding import raiseIfNoneGitHubIssueNumber3, The
from mapFolding.someAssemblyRequired import (
	ast_Identifier,
	astModuleToIngredientsFunction,
	Be,
	DOT,
	extractFunctionDef,
	Grab,
	IfThis,
	IngredientsFunction,
	IngredientsModule,
	inlineFunctionDef,
	LedgerOfImports,
	Make,
	NodeChanger,
	NodeTourist,
	parseLogicalPath2astModule,
	parsePathFilename2astModule,
	removeUnusedParameters,
	str_nameDOTname,
	Then,
	write_astModule,
	DeReConstructField2ast,
	ShatteredDataclass,
)
from mapFolding.someAssemblyRequired.toolkitNumba import decorateCallableWithNumba, parametersNumbaLight
from mapFolding.someAssemblyRequired.transformationTools import (
	removeDataclassFromFunction,
	shatter_dataclassesDOTdataclass,
	unpackDataclassCallFunctionRepackDataclass,
)
from pathlib import PurePath
from Z0Z_tools import importLogicalPath2Callable
import ast
import dataclasses

algorithmSourceModuleHARDCODED = 'daoOfMapFolding'
sourceCallableIdentifierHARDCODED = 'count'
logicalPathInfixHARDCODED: ast_Identifier = 'syntheticModules'
theCountingIdentifierHARDCODED: ast_Identifier = 'groupsOfFolds'
dataPackingModuleIdentifierHARDCODED: ast_Identifier = 'dataPacking'

def makeInitializeGroupsOfFolds() -> None:
	callableIdentifierHARDCODED = 'initializeGroupsOfFolds'
	moduleIdentifierHARDCODED: ast_Identifier = 'initializeCount'

	algorithmSourceModule = algorithmSourceModuleHARDCODED
	sourceCallableIdentifier = sourceCallableIdentifierHARDCODED
	logicalPathSourceModule = '.'.join([The.packageName, algorithmSourceModule])

	callableIdentifier = callableIdentifierHARDCODED
	logicalPathInfix = logicalPathInfixHARDCODED
	moduleIdentifier = moduleIdentifierHARDCODED

	astModule = parseLogicalPath2astModule(logicalPathSourceModule)
	countInitializeIngredients = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule)
		, LedgerOfImports(astModule))

	countInitializeIngredients.astFunctionDef.name = callableIdentifier

	dataclassInstanceIdentifier = NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(countInitializeIngredients.astFunctionDef)
	if dataclassInstanceIdentifier is None: raise raiseIfNoneGitHubIssueNumber3
	theCountingIdentifier = theCountingIdentifierHARDCODED

	findThis = IfThis.isWhileAttributeNamespace_IdentifierGreaterThan0(dataclassInstanceIdentifier, 'leaf1ndex')
	doThat = Grab.testAttribute(Grab.andDoAllOf([ Grab.opsAttribute(Then.replaceWith([ast.Eq()])), Grab.leftAttribute(Grab.attrAttribute(Then.replaceWith(theCountingIdentifier))) ])) # type: ignore
	NodeChanger(findThis, doThat).visit(countInitializeIngredients.astFunctionDef.body[0])

	ingredientsModule = IngredientsModule(countInitializeIngredients)

	pathFilename = PurePath(The.pathPackage, logicalPathInfix, moduleIdentifier + The.fileExtension)

	write_astModule(ingredientsModule, pathFilename, The.packageName)

def makeDaoOfMapFolding() -> PurePath:
	moduleIdentifierHARDCODED: ast_Identifier = 'daoOfMapFolding'

	algorithmSourceModule = algorithmSourceModuleHARDCODED
	sourceCallableIdentifier = sourceCallableIdentifierHARDCODED
	logicalPathSourceModule = '.'.join([The.packageName, algorithmSourceModule])

	logicalPathInfix = logicalPathInfixHARDCODED
	moduleIdentifier = moduleIdentifierHARDCODED

	astModule = parseLogicalPath2astModule(logicalPathSourceModule)
	daoOfMapFolding = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule)
		, LedgerOfImports(astModule))

	dataclassName: ast.expr | None = NodeTourist(Be.arg, Then.extractIt(DOT.annotation)).captureLastMatch(daoOfMapFolding.astFunctionDef)
	if dataclassName is None: raise raiseIfNoneGitHubIssueNumber3
	dataclass_Identifier: ast_Identifier | None = NodeTourist(Be.Name, Then.extractIt(DOT.id)).captureLastMatch(dataclassName)
	if dataclass_Identifier is None: raise raiseIfNoneGitHubIssueNumber3

	dataclassLogicalPathModule = None
	for moduleWithLogicalPath, listNameTuples in daoOfMapFolding.imports.dictionaryImportFrom.items():
		for nameTuple in listNameTuples:
			if nameTuple[0] == dataclass_Identifier:
				dataclassLogicalPathModule = moduleWithLogicalPath
				break
		if dataclassLogicalPathModule:
			break
	if dataclassLogicalPathModule is None: raise raiseIfNoneGitHubIssueNumber3
	dataclassInstanceIdentifier = NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(daoOfMapFolding.astFunctionDef)
	if dataclassInstanceIdentifier is None: raise raiseIfNoneGitHubIssueNumber3
	shatteredDataclass = shatter_dataclassesDOTdataclass(dataclassLogicalPathModule, dataclass_Identifier, dataclassInstanceIdentifier)

	# theCountingIdentifier = theCountingIdentifierHARDCODED
	# doubleTheCount = Make.AugAssign(Make.Attribute(ast.Name(dataclassInstanceIdentifier), theCountingIdentifier), ast.Mult(), Make.Constant(2))
	# findThis = be.Return
	# doThat = Then.insertThisAbove([doubleTheCount])
	# NodeChanger(findThis, doThat).visit(daoOfMapFolding.astFunctionDef)

	daoOfMapFolding.imports.update(shatteredDataclass.imports)
	daoOfMapFolding = removeDataclassFromFunction(daoOfMapFolding, shatteredDataclass)

	daoOfMapFolding = removeUnusedParameters(daoOfMapFolding)

	daoOfMapFolding = decorateCallableWithNumba(daoOfMapFolding, parametersNumbaLight)

	sourceCallableIdentifier = The.sourceCallableDispatcher

	doTheNeedful: IngredientsFunction = astModuleToIngredientsFunction(astModule, sourceCallableIdentifier)
	doTheNeedful.imports.update(shatteredDataclass.imports)
	targetCallableIdentifier = daoOfMapFolding.astFunctionDef.name
	doTheNeedful = unpackDataclassCallFunctionRepackDataclass(doTheNeedful, targetCallableIdentifier, shatteredDataclass)
	astTuple: ast.Tuple | None = NodeTourist(Be.Return, Then.extractIt(DOT.value)).captureLastMatch(daoOfMapFolding.astFunctionDef) # type: ignore
	if astTuple is None: raise raiseIfNoneGitHubIssueNumber3
	astTuple.ctx = ast.Store()
	ast.Return()

	findThis = ClassIsAndAttribute.valueIs(ast.Assign, IfThis.isCall_Identifier(targetCallableIdentifier))
	doThat = Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(targetCallableIdentifier), astTuple.elts)))
	changeAssignCallToTarget = NodeChanger(findThis, doThat)
	changeAssignCallToTarget.visit(doTheNeedful.astFunctionDef)

	ingredientsModule = IngredientsModule([daoOfMapFolding, doTheNeedful])
	ingredientsModule.removeImportFromModule('numpy')

	pathFilename = PurePath(The.pathPackage, logicalPathInfix, moduleIdentifier + The.fileExtension)

	write_astModule(ingredientsModule, pathFilename, The.packageName)

	return pathFilename

def makeDaoOfMapFoldingParallel() -> PurePath:
	moduleIdentifierHARDCODED: ast_Identifier = 'countParallel'

	algorithmSourceModule = algorithmSourceModuleHARDCODED
	sourceCallableIdentifier = sourceCallableIdentifierHARDCODED
	logicalPathSourceModule = '.'.join([The.packageName, algorithmSourceModule])

	logicalPathInfix = logicalPathInfixHARDCODED
	moduleIdentifier = moduleIdentifierHARDCODED

	astModule = parseLogicalPath2astModule(logicalPathSourceModule)
	ingredientsFunction = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule), LedgerOfImports(astModule))

	dataclassName: ast.expr | None = NodeTourist(Be.arg, Then.extractIt(DOT.annotation)).captureLastMatch(ingredientsFunction.astFunctionDef)
	if dataclassName is None: raise raiseIfNoneGitHubIssueNumber3
	dataclass_Identifier: ast_Identifier | None = NodeTourist(Be.Name, Then.extractIt(DOT.id)).captureLastMatch(dataclassName)
	if dataclass_Identifier is None: raise raiseIfNoneGitHubIssueNumber3

	dataclassLogicalPathModule = None
	for moduleWithLogicalPath, listNameTuples in ingredientsFunction.imports.dictionaryImportFrom.items():
		for nameTuple in listNameTuples:
			if nameTuple[0] == dataclass_Identifier:
				dataclassLogicalPathModule = moduleWithLogicalPath
				break
		if dataclassLogicalPathModule:
			break
	if dataclassLogicalPathModule is None: raise raiseIfNoneGitHubIssueNumber3
	dataclassInstanceIdentifier = NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef)
	if dataclassInstanceIdentifier is None: raise raiseIfNoneGitHubIssueNumber3
	shatteredDataclass = shatter_dataclassesDOTdataclass(dataclassLogicalPathModule, dataclass_Identifier, dataclassInstanceIdentifier)

	# Start add the parallel state fields to the count function ================================================
	dataclassBaseFields = dataclasses.fields(importLogicalPath2Callable(dataclassLogicalPathModule, dataclass_Identifier))  # pyright: ignore [reportArgumentType]
	dataclass_IdentifierParallel = 'Parallel' + dataclass_Identifier
	dataclassFieldsParallel = dataclasses.fields(importLogicalPath2Callable(dataclassLogicalPathModule, dataclass_IdentifierParallel))  # pyright: ignore [reportArgumentType]
	onlyParallelFields = [field for field in dataclassFieldsParallel if field.name not in [fieldBase.name for fieldBase in dataclassBaseFields]]

	Official_fieldOrder: list[ast_Identifier] = []
	dictionaryDeReConstruction: dict[ast_Identifier, DeReConstructField2ast] = {}

	dataclassClassDef = extractClassDef(parseLogicalPath2astModule(dataclassLogicalPathModule), dataclass_IdentifierParallel)
	if not isinstance(dataclassClassDef, ast.ClassDef): raise ValueError(f"I could not find `{dataclass_IdentifierParallel = }` in `{dataclassLogicalPathModule = }`.")

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
	shatteredDataclassParallel.repack = Make.Assign([Make.Name(dataclassInstanceIdentifier)], value=Make.Call(Make.Name(dataclass_IdentifierParallel), list_keyword=shatteredDataclassParallel.list_keyword_field__field4init))
	shatteredDataclassParallel.signatureReturnAnnotation = Make.Subscript(Make.Name('tuple'), Make.Tuple(shatteredDataclassParallel.listAnnotations))

	shatteredDataclassParallel.imports.update(*(dictionaryDeReConstruction[field].ledger for field in Official_fieldOrder))
	shatteredDataclassParallel.imports.addImportFrom_asStr(dataclassLogicalPathModule, dataclass_IdentifierParallel)
	shatteredDataclassParallel.imports.update(shatteredDataclass.imports)
	shatteredDataclassParallel.imports.removeImportFrom(dataclassLogicalPathModule, dataclass_Identifier)

	# End add the parallel state fields to the count function ================================================

	ingredientsFunction.imports.update(shatteredDataclassParallel.imports)
	ingredientsFunction = removeDataclassFromFunction(ingredientsFunction, shatteredDataclassParallel)

	# Start add the parallel logic to the count function ================================================

	findThis = ClassIsAndAttribute.testIs(ast.While, ClassIsAndAttribute.leftIs(ast.Compare, IfThis.isName_Identifier('leafConnectee')))
	doThat = Then.extractIt(DOT.body)
	captureCountGapsCodeBlock: NodeTourist[ast.While, Sequence[ast.stmt]] = NodeTourist(findThis, doThat)
	countGapsCodeBlock = captureCountGapsCodeBlock.captureLastMatch(ingredientsFunction.astFunctionDef)
	if countGapsCodeBlock is None: raise raiseIfNoneGitHubIssueNumber3

	thisIsMyTaskIndexCodeBlock = ast.If(ast.BoolOp(ast.Or()
		, values=[ast.Compare(ast.Name('leaf1ndex'), ops=[ast.NotEq()], comparators=[ast.Name('taskDivisions')])
				, ast.Compare(ast.BinOp(ast.Name('leafConnectee'), op=ast.Mod(), right=ast.Name('taskDivisions')), ops=[ast.Eq()], comparators=[ast.Name('taskIndex')])])
	, body=list(countGapsCodeBlock[0:-1]))

	countGapsCodeBlockNew: list[ast.stmt] = [thisIsMyTaskIndexCodeBlock, countGapsCodeBlock[-1]]

	doThat = Grab.bodyAttribute(Then.replaceWith(countGapsCodeBlockNew))
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef)

	# End add the parallel logic to the count function ================================================

	ingredientsFunction = removeUnusedParameters(ingredientsFunction)

	ingredientsFunction = decorateCallableWithNumba(ingredientsFunction, parametersNumbaLight)

	# Start unpack/repack the dataclass function ================================================
	sourceCallableIdentifier = The.sourceCallableDispatcher

	unRepackDataclass: IngredientsFunction = astModuleToIngredientsFunction(astModule, sourceCallableIdentifier)
	unRepackDataclass.astFunctionDef.name = 'unRepack' + dataclass_IdentifierParallel
	unRepackDataclass.imports.update(shatteredDataclassParallel.imports)
	findThis = ClassIsAndAttribute.annotationIs(ast.arg, IfThis.isName_Identifier(dataclass_Identifier)) # type: ignore
	doThat = Grab.annotationAttribute(Grab.idAttribute(Then.replaceWith(dataclass_IdentifierParallel))) # type: ignore
	NodeChanger(findThis, doThat).visit(unRepackDataclass.astFunctionDef) # type: ignore
	unRepackDataclass.astFunctionDef.returns = Make.Name(dataclass_IdentifierParallel)
	targetCallableIdentifier = ingredientsFunction.astFunctionDef.name
	unRepackDataclass = unpackDataclassCallFunctionRepackDataclass(unRepackDataclass, targetCallableIdentifier, shatteredDataclassParallel)

	astTuple: ast.Tuple | None = NodeTourist(Be.Return, Then.extractIt(DOT.value)).captureLastMatch(ingredientsFunction.astFunctionDef) # type: ignore
	if astTuple is None: raise raiseIfNoneGitHubIssueNumber3
	astTuple.ctx = ast.Store()
	findThis = ClassIsAndAttribute.valueIs(ast.Assign, IfThis.isCall_Identifier(targetCallableIdentifier))
	doThat = Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(targetCallableIdentifier), astTuple.elts)))
	changeAssignCallToTarget = NodeChanger(findThis, doThat)
	changeAssignCallToTarget.visit(unRepackDataclass.astFunctionDef)

	ingredientsDoTheNeedful: IngredientsFunction = IngredientsFunction(
		astFunctionDef = ast.FunctionDef(name='doTheNeedful'
			, args=ast.arguments(args=[ast.arg('state', annotation=ast.Name(dataclass_IdentifierParallel)), ast.arg('concurrencyLimit', annotation=ast.Name('int'))])
			, body=[ast.Assign(targets=[ast.Name('stateParallel', ctx=ast.Store())], value=ast.Call(func=ast.Name('deepcopy'), args=[ast.Name('state')]))
				, ast.AnnAssign(target=ast.Name('listStatesParallel', ctx=ast.Store()), annotation=ast.Subscript(value=ast.Name('list'), slice=ast.Name(dataclass_IdentifierParallel)), value=ast.BinOp(left=ast.List(elts=[ast.Name('stateParallel')]), op=ast.Mult(), right=ast.Attribute(value=ast.Name('stateParallel'), attr='taskDivisions')), simple=1)
				, ast.AnnAssign(target=ast.Name('groupsOfFoldsTotal', ctx=ast.Store()), annotation=ast.Name('int'), value=ast.Constant(value=0), simple=1)

				, ast.AnnAssign(target=ast.Name('dictionaryConcurrency', ctx=ast.Store()), annotation=ast.Subscript(value=ast.Name('dict'), slice=ast.Tuple(elts=[ast.Name('int'), ast.Subscript(value=ast.Name('ConcurrentFuture'), slice=ast.Name(dataclass_IdentifierParallel))])), value=ast.Dict(), simple=1)
				, ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Name('ProcessPoolExecutor'), args=[ast.Name('concurrencyLimit')]), optional_vars=ast.Name('concurrencyManager', ctx=ast.Store()))]
					, body=[ast.For(target=ast.Name('indexSherpa', ctx=ast.Store()), iter=ast.Call(func=ast.Name('range'), args=[ast.Attribute(value=ast.Name('stateParallel'), attr='taskDivisions')])
							, body=[ast.Assign(targets=[ast.Name('state', ctx=ast.Store())], value=ast.Call(func=ast.Name('deepcopy'), args=[ast.Name('stateParallel')]))
								, ast.Assign(targets=[ast.Attribute(value=ast.Name('state'), attr='taskIndex', ctx=ast.Store())], value=ast.Name('indexSherpa'))
								, ast.Assign(targets=[ast.Subscript(value=ast.Name('dictionaryConcurrency'), slice=ast.Name('indexSherpa'), ctx=ast.Store())], value=ast.Call(func=ast.Attribute(value=ast.Name('concurrencyManager'), attr='submit'), args=[ast.Name(unRepackDataclass.astFunctionDef.name), ast.Name('state')]))])
						, ast.For(target=ast.Name('indexSherpa', ctx=ast.Store()), iter=ast.Call(func=ast.Name('range'), args=[ast.Attribute(value=ast.Name('stateParallel'), attr='taskDivisions')])
							, body=[ast.Assign(targets=[ast.Subscript(value=ast.Name('listStatesParallel'), slice=ast.Name('indexSherpa'), ctx=ast.Store())], value=ast.Call(func=ast.Attribute(value=ast.Subscript(value=ast.Name('dictionaryConcurrency'), slice=ast.Name('indexSherpa')), attr='result')))
								, ast.AugAssign(target=ast.Name('groupsOfFoldsTotal', ctx=ast.Store()), op=ast.Add(), value=ast.Attribute(value=ast.Subscript(value=ast.Name('listStatesParallel'), slice=ast.Name('indexSherpa')), attr='groupsOfFolds'))])])

				, ast.AnnAssign(target=ast.Name('foldsTotal', ctx=ast.Store()), annotation=ast.Name('int'), value=ast.BinOp(left=ast.Name('groupsOfFoldsTotal'), op=ast.Mult(), right=ast.Attribute(value=ast.Name('stateParallel'), attr='leavesTotal')), simple=1)
				, ast.Return(value=ast.Tuple(elts=[ast.Name('foldsTotal'), ast.Name('listStatesParallel')]))]
			, returns=ast.Subscript(value=ast.Name('tuple'), slice=ast.Tuple(elts=[ast.Name('int'), ast.Subscript(value=ast.Name('list'), slice=ast.Name(dataclass_IdentifierParallel))])))
		, imports = LedgerOfImports(Make.Module([ast.ImportFrom(module='concurrent.futures', names=[ast.alias(name='Future', asname='ConcurrentFuture'), ast.alias(name='ProcessPoolExecutor')], level=0),
			ast.ImportFrom(module='copy', names=[ast.alias(name='deepcopy')], level=0),
			ast.ImportFrom(module='multiprocessing', names=[ast.alias(name='set_start_method', asname='multiprocessing_set_start_method')], level=0),])
		)
	)

	ingredientsModule = IngredientsModule([ingredientsFunction, unRepackDataclass, ingredientsDoTheNeedful]
						, prologue = Make.Module([ast.If(test=ast.Compare(left=ast.Name('__name__'), ops=[ast.Eq()], comparators=[ast.Constant(value='__main__')]), body=[ast.Expr(value=ast.Call(func=ast.Name('multiprocessing_set_start_method'), args=[ast.Constant(value='spawn')]))])])
	)
	ingredientsModule.removeImportFromModule('numpy')

	pathFilename = PurePath(The.pathPackage, logicalPathInfix, moduleIdentifier + The.fileExtension)

	write_astModule(ingredientsModule, pathFilename, The.packageName)
	return pathFilename

def makeTheorem2() -> PurePath:
	moduleIdentifierHARDCODED: ast_Identifier = 'theorem2'

	algorithmSourceModule = algorithmSourceModuleHARDCODED
	sourceCallableIdentifier = sourceCallableIdentifierHARDCODED
	logicalPathSourceModule = '.'.join([The.packageName, algorithmSourceModule])

	logicalPathInfix = logicalPathInfixHARDCODED
	moduleIdentifier = moduleIdentifierHARDCODED

	astModule = parseLogicalPath2astModule(logicalPathSourceModule)
	countTheorem2 = IngredientsFunction(inlineFunctionDef(sourceCallableIdentifier, astModule)
		, LedgerOfImports(astModule))

	dataclassInstanceIdentifier = NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(countTheorem2.astFunctionDef)
	if dataclassInstanceIdentifier is None: raise raiseIfNoneGitHubIssueNumber3

	findThis = IfThis.isWhileAttributeNamespace_IdentifierGreaterThan0(dataclassInstanceIdentifier, 'leaf1ndex')
	doThat = Grab.testAttribute(Grab.comparatorsAttribute(Then.replaceWith([Make.Constant(4)]))) # type: ignore
	NodeChanger(findThis, doThat).visit(countTheorem2.astFunctionDef)

	findThis = IfThis.isIfAttributeNamespace_IdentifierGreaterThan0(dataclassInstanceIdentifier, 'leaf1ndex')
	doThat = Then.extractIt(DOT.body)
	insertLeaf = NodeTourist(findThis, doThat).captureLastMatch(countTheorem2.astFunctionDef)
	findThis = IfThis.isIfAttributeNamespace_IdentifierGreaterThan0(dataclassInstanceIdentifier, 'leaf1ndex')
	doThat = Then.replaceWith(insertLeaf)
	NodeChanger(findThis, doThat).visit(countTheorem2.astFunctionDef)

	findThis = IfThis.isAttributeNamespace_IdentifierGreaterThan0(dataclassInstanceIdentifier, 'leaf1ndex')
	doThat = Then.removeIt
	NodeChanger(findThis, doThat).visit(countTheorem2.astFunctionDef)

	findThis = IfThis.isAttributeNamespace_IdentifierLessThanOrEqual0(dataclassInstanceIdentifier, 'leaf1ndex')
	doThat = Then.removeIt
	NodeChanger(findThis, doThat).visit(countTheorem2.astFunctionDef)

	theCountingIdentifier = theCountingIdentifierHARDCODED
	doubleTheCount = Make.AugAssign(Make.Attribute(ast.Name(dataclassInstanceIdentifier), theCountingIdentifier), ast.Mult(), Make.Constant(2))
	findThis = Be.Return
	doThat = Then.insertThisAbove([doubleTheCount])
	NodeChanger(findThis, doThat).visit(countTheorem2.astFunctionDef)

	ingredientsModule = IngredientsModule(countTheorem2)

	pathFilename = PurePath(The.pathPackage, logicalPathInfix, moduleIdentifier + The.fileExtension)

	write_astModule(ingredientsModule, pathFilename, The.packageName)

	return pathFilename

def trimTheorem2(pathFilenameSource: PurePath) -> PurePath:
	logicalPathInfix = logicalPathInfixHARDCODED
	sourceCallableIdentifier = sourceCallableIdentifierHARDCODED
	ingredientsFunction = astModuleToIngredientsFunction(parsePathFilename2astModule(pathFilenameSource), sourceCallableIdentifier)

	dataclassInstanceIdentifier = NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef)
	if dataclassInstanceIdentifier is None: raise raiseIfNoneGitHubIssueNumber3

	findThis = IfThis.isIfUnaryNotAttributeNamespace_Identifier(dataclassInstanceIdentifier, 'dimensionsUnconstrained')
	doThat = Then.removeIt
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef)

	ingredientsModule = IngredientsModule(ingredientsFunction)
	ingredientsModule.removeImportFromModule('numpy')

	pathFilename = pathFilenameSource.with_stem(pathFilenameSource.stem + 'Trimmed')

	write_astModule(ingredientsModule, pathFilename, The.packageName)

	logicalPath: list[str] = []
	if The.packageName:
		logicalPath.append(The.packageName)
	if logicalPathInfix:
		logicalPath.append(logicalPathInfix)
	logicalPath.append(pathFilename.stem)
	moduleWithLogicalPath: str_nameDOTname = '.'.join(logicalPath)

	astImportFrom: ast.ImportFrom = Make.ImportFrom(moduleWithLogicalPath, list_alias=[Make.alias(ingredientsFunction.astFunctionDef.name)])

	return pathFilename

def numbaOnTheorem2(pathFilenameSource: PurePath) -> ast.ImportFrom:
	logicalPathInfix = logicalPathInfixHARDCODED
	sourceCallableIdentifier = sourceCallableIdentifierHARDCODED
	countNumbaTheorem2 = astModuleToIngredientsFunction(parsePathFilename2astModule(pathFilenameSource), sourceCallableIdentifier)
	dataclassName: ast.expr | None = NodeTourist(Be.arg, Then.extractIt(DOT.annotation)).captureLastMatch(countNumbaTheorem2.astFunctionDef)
	if dataclassName is None: raise raiseIfNoneGitHubIssueNumber3
	dataclass_Identifier: ast_Identifier | None = NodeTourist(Be.Name, Then.extractIt(DOT.id)).captureLastMatch(dataclassName)
	if dataclass_Identifier is None: raise raiseIfNoneGitHubIssueNumber3

	dataclassLogicalPathModule = None
	for moduleWithLogicalPath, listNameTuples in countNumbaTheorem2.imports.dictionaryImportFrom.items():
		for nameTuple in listNameTuples:
			if nameTuple[0] == dataclass_Identifier:
				dataclassLogicalPathModule = moduleWithLogicalPath
				break
		if dataclassLogicalPathModule:
			break
	if dataclassLogicalPathModule is None: raise raiseIfNoneGitHubIssueNumber3
	dataclassInstanceIdentifier = NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(countNumbaTheorem2.astFunctionDef)
	if dataclassInstanceIdentifier is None: raise raiseIfNoneGitHubIssueNumber3
	shatteredDataclass = shatter_dataclassesDOTdataclass(dataclassLogicalPathModule, dataclass_Identifier, dataclassInstanceIdentifier)

	countNumbaTheorem2.imports.update(shatteredDataclass.imports)
	countNumbaTheorem2 = removeDataclassFromFunction(countNumbaTheorem2, shatteredDataclass)

	countNumbaTheorem2 = removeUnusedParameters(countNumbaTheorem2)

	countNumbaTheorem2 = decorateCallableWithNumba(countNumbaTheorem2, parametersNumbaLight)

	ingredientsModule = IngredientsModule(countNumbaTheorem2)
	ingredientsModule.removeImportFromModule('numpy')

	pathFilename = pathFilenameSource.with_stem(pathFilenameSource.stem.replace('Trimmed', '') + 'Numba')

	write_astModule(ingredientsModule, pathFilename, The.packageName)

	logicalPath: list[str] = []
	if The.packageName:
		logicalPath.append(The.packageName)
	if logicalPathInfix:
		logicalPath.append(logicalPathInfix)
	logicalPath.append(pathFilename.stem)
	moduleWithLogicalPath: str_nameDOTname = '.'.join(logicalPath)

	astImportFrom: ast.ImportFrom = Make.ImportFrom(moduleWithLogicalPath, list_alias=[Make.alias(countNumbaTheorem2.astFunctionDef.name)])

	return astImportFrom

def makeUnRePackDataclass(astImportFrom: ast.ImportFrom) -> None:
	callableIdentifierHARDCODED: ast_Identifier = 'sequential'

	algorithmSourceModule = algorithmSourceModuleHARDCODED
	sourceCallableIdentifier = The.sourceCallableDispatcher
	logicalPathSourceModule = '.'.join([The.packageName, algorithmSourceModule])

	logicalPathInfix = logicalPathInfixHARDCODED
	moduleIdentifier = dataPackingModuleIdentifierHARDCODED
	callableIdentifier = callableIdentifierHARDCODED

	ingredientsFunction: IngredientsFunction = astModuleToIngredientsFunction(parseLogicalPath2astModule(logicalPathSourceModule), sourceCallableIdentifier)
	ingredientsFunction.astFunctionDef.name = callableIdentifier
	dataclassName: ast.expr | None = NodeTourist(Be.arg, Then.extractIt(DOT.annotation)).captureLastMatch(ingredientsFunction.astFunctionDef)
	if dataclassName is None: raise raiseIfNoneGitHubIssueNumber3
	dataclass_Identifier: ast_Identifier | None = NodeTourist(Be.Name, Then.extractIt(DOT.id)).captureLastMatch(dataclassName)
	if dataclass_Identifier is None: raise raiseIfNoneGitHubIssueNumber3

	dataclassLogicalPathModule = None
	for moduleWithLogicalPath, listNameTuples in ingredientsFunction.imports.dictionaryImportFrom.items():
		for nameTuple in listNameTuples:
			if nameTuple[0] == dataclass_Identifier:
				dataclassLogicalPathModule = moduleWithLogicalPath
				break
		if dataclassLogicalPathModule:
			break
	if dataclassLogicalPathModule is None: raise raiseIfNoneGitHubIssueNumber3
	dataclassInstanceIdentifier = NodeTourist(Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef)
	if dataclassInstanceIdentifier is None: raise raiseIfNoneGitHubIssueNumber3
	shatteredDataclass = shatter_dataclassesDOTdataclass(dataclassLogicalPathModule, dataclass_Identifier, dataclassInstanceIdentifier)

	ingredientsFunction.imports.update(shatteredDataclass.imports)
	ingredientsFunction.imports.addAst(astImportFrom)
	targetCallableIdentifier = astImportFrom.names[0].name
	ingredientsFunction = unpackDataclassCallFunctionRepackDataclass(ingredientsFunction, targetCallableIdentifier, shatteredDataclass)
	if astImportFrom.module is None: raise raiseIfNoneGitHubIssueNumber3
	targetFunctionDef = extractFunctionDef(parseLogicalPath2astModule(astImportFrom.module), targetCallableIdentifier)
	if targetFunctionDef is None: raise raiseIfNoneGitHubIssueNumber3
	astTuple: ast.Tuple | None = NodeTourist(Be.Return, Then.extractIt(DOT.value)).captureLastMatch(targetFunctionDef) # type: ignore
	if astTuple is None: raise raiseIfNoneGitHubIssueNumber3
	astTuple.ctx = ast.Store()

	findThis = ClassIsAndAttribute.valueIs(ast.Assign, IfThis.isCall_Identifier(targetCallableIdentifier))
	doThat = Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(targetCallableIdentifier), astTuple.elts)))
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef)

	ingredientsModule = IngredientsModule(ingredientsFunction)
	ingredientsModule.removeImportFromModule('numpy')

	pathFilename = PurePath(The.pathPackage, logicalPathInfix, moduleIdentifier + The.fileExtension)

	write_astModule(ingredientsModule, pathFilename, The.packageName)

if __name__ == '__main__':
	makeInitializeGroupsOfFolds()
	pathFilename = makeTheorem2()
	pathFilename = trimTheorem2(pathFilename)
	astImportFrom = numbaOnTheorem2(pathFilename)
	makeUnRePackDataclass(astImportFrom)
	pathFilename = makeDaoOfMapFolding()
	makeDaoOfMapFoldingParallel()
