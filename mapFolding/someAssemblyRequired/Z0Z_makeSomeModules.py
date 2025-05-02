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
	astTuple: ast.Tuple | None = NodeTourist(Be.Return, Then.extractIt(DOT.value)).captureLastMatch(daoOfMapFolding.astFunctionDef)
	if astTuple is None: raise raiseIfNoneGitHubIssueNumber3
	astTuple.ctx = ast.Store()

	findThis = ClassIsAndAttribute.valueIs(ast.Assign, IfThis.isCall_Identifier(targetCallableIdentifier))
	doThat = Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(targetCallableIdentifier), astTuple.elts)))
	changeAssignCallToTarget = NodeChanger(findThis, doThat)
	changeAssignCallToTarget.visit(doTheNeedful.astFunctionDef)

	ingredientsModule = IngredientsModule([daoOfMapFolding, doTheNeedful])
	ingredientsModule.removeImportFromModule('numpy')

	pathFilename = PurePath(The.pathPackage, logicalPathInfix, moduleIdentifier + The.fileExtension)

	write_astModule(ingredientsModule, pathFilename, The.packageName)

	return pathFilename

def makeDaoOfMapFoldingParallel(pathFilenameSource: PurePath) -> PurePath:
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

def makeDaoOfMapFoldingParallelV1() -> PurePath:
	"""Notes
	Additional state information: taskDivisions, taskIndex
	Additional flow information: CPUlimit (already handled in basecamp.py)
	Additional count logic: `if thisIsMyTaskIndex`

	Make a state container to store the state of one task.
	Make a state container to store the state of all tasks.

	I know you don't want to do this, but if you segregate the parallel computations now, they will have their own
	transformation assembly line--that you don't have to care about.
	"""
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

	findThis = IfThis.isIfUnaryNotAttributeNamespace_Identifier(dataclassInstanceIdentifier, 'dimensionsUnconstrained')
	doThat = Then.removeIt
	NodeChanger(findThis, doThat).visit(ingredientsFunction.astFunctionDef)

	ingredientsFunction.imports.update(shatteredDataclass.imports)
	ingredientsFunction = removeDataclassFromFunction(ingredientsFunction, shatteredDataclass)

	# Start add the parallel state fields to the count function ================================================
	dataclassBaseFields = dataclasses.fields(importLogicalPath2Callable(dataclassLogicalPathModule, dataclass_Identifier))  # pyright: ignore [reportArgumentType]
	dataclass_IdentifierParallel = 'Parallel' + dataclass_Identifier
	dataclassFieldsParallel = dataclasses.fields(importLogicalPath2Callable(dataclassLogicalPathModule, dataclass_IdentifierParallel))  # pyright: ignore [reportArgumentType]
	onlyParallelFields = [field for field in dataclassFieldsParallel if field.name not in [fieldBase.name for fieldBase in dataclassBaseFields]]

	Official_fieldOrder: list[ast_Identifier] = []
	dictionaryDeReConstruction: dict[ast_Identifier, DeReConstructField2ast] = {}

	dataclassClassDef = extractClassDef(parseLogicalPath2astModule(dataclassLogicalPathModule), dataclass_IdentifierParallel)
	if not isinstance(dataclassClassDef, ast.ClassDef): raise ValueError(f"I could not find `{dataclass_IdentifierParallel = }` in `{dataclassLogicalPathModule = }`.")

	countingVariable = None
	for aField in onlyParallelFields:
		Official_fieldOrder.append(aField.name)
		dictionaryDeReConstruction[aField.name] = DeReConstructField2ast(dataclassLogicalPathModule, dataclassClassDef, dataclassInstanceIdentifier, aField)

	shatteredDataclassParallel = ShatteredDataclass(
		countingVariableAnnotation=dictionaryDeReConstruction[countingVariable].astAnnotation if countingVariable else None,
		countingVariableName=dictionaryDeReConstruction[countingVariable].astName if countingVariable else None,
		field2AnnAssign={dictionaryDeReConstruction[field].name: dictionaryDeReConstruction[field].astAnnAssignConstructor for field in Official_fieldOrder},
		Z0Z_field2AnnAssign={dictionaryDeReConstruction[field].name: dictionaryDeReConstruction[field].Z0Z_hack for field in Official_fieldOrder},
		list_argAnnotated4ArgumentsSpecification=[dictionaryDeReConstruction[field].ast_argAnnotated for field in Official_fieldOrder],
		list_keyword_field__field4init=[dictionaryDeReConstruction[field].ast_keyword_field__field for field in Official_fieldOrder if dictionaryDeReConstruction[field].init],
		listAnnotations=[dictionaryDeReConstruction[field].astAnnotation for field in Official_fieldOrder],
		listName4Parameters=[dictionaryDeReConstruction[field].astName for field in Official_fieldOrder],
		listUnpack=[Make.AnnAssign(dictionaryDeReConstruction[field].astName, dictionaryDeReConstruction[field].astAnnotation, dictionaryDeReConstruction[field].ast_nameDOTname) for field in Official_fieldOrder],
		map_stateDOTfield2Name={dictionaryDeReConstruction[field].ast_nameDOTname: dictionaryDeReConstruction[field].astName for field in Official_fieldOrder},
		)
	shatteredDataclassParallel.fragments4AssignmentOrParameters = Make.Tuple(shatteredDataclassParallel.listName4Parameters, ast.Store())
	fragments4AssignmentOrParametersExtended = Make.Tuple(shatteredDataclass.listName4Parameters + shatteredDataclassParallel.listName4Parameters, ast.Store())
	shatteredDataclassParallel.repack = Make.Assign([Make.Name(dataclassInstanceIdentifier)], value=Make.Call(Make.Name(dataclass_IdentifierParallel), list_keyword=shatteredDataclassParallel.list_keyword_field__field4init))

	shatteredDataclassParallel.signatureReturnAnnotation = Make.Subscript(Make.Name('tuple'), Make.Tuple(shatteredDataclassParallel.listAnnotations))
	signatureReturnAnnotationExtended = Make.Subscript(Make.Name('tuple'), Make.Tuple(shatteredDataclass.listAnnotations + shatteredDataclassParallel.listAnnotations))
	shatteredDataclassParallel.imports.update(*(dictionaryDeReConstruction[field].ledger for field in Official_fieldOrder))
	shatteredDataclassParallel.imports.addImportFrom_asStr(dataclassLogicalPathModule, dataclass_IdentifierParallel)

	ingredientsFunction.astFunctionDef.args.args.extend(shatteredDataclassParallel.list_argAnnotated4ArgumentsSpecification)
	ingredientsFunction.astFunctionDef.returns = signatureReturnAnnotationExtended
	changeReturnCallable = NodeChanger(Be.Return, Then.replaceWith(Make.Return(fragments4AssignmentOrParametersExtended)))
	changeReturnCallable.visit(ingredientsFunction.astFunctionDef)
	# End add the parallel state fields to the count function ================================================
	# Start add the parallel logic to the count function ================================================

	"""I need to do the equivalent of this:
	This:
	state = countGaps(state)
	Becomes:
	if thisIsMyTaskIndex(state):
		state = countGaps(state)

	def thisIsMyTaskIndex(state: ComputationState) -> bool:
		return (state.leaf1ndex != state.taskDivisions) or (state.leafConnectee % state.taskDivisions == state.taskIndex)


	It _might_ be easier to add the conditional check before inlining the functions.
	But, even if it is easier, I think I want to focus on making a highly reusable function to transform sequential count to parallel count.
	omg, that sounds hard.
	I'll come back to this.
	def countGaps(state: ComputationState) -> ComputationState:
		state.gapsWhere[state.gap1ndexCeiling] = state.leafConnectee
		if state.countDimensionsGapped[state.leafConnectee] == 0:
			state = incrementGap1ndexCeiling(state)
		state.countDimensionsGapped[state.leafConnectee] += 1
		return state

	while leafConnectee != leaf1ndex:
	"""
	findThis = IfThis.isWhileAttributeNamespace_IdentifierGreaterThan0
	# End add the parallel logic to the count function ================================================

	ingredientsFunction = removeUnusedParameters(ingredientsFunction)

	ingredientsFunction = decorateCallableWithNumba(ingredientsFunction, parametersNumbaLight)

	sourceCallableIdentifier = The.sourceCallableDispatcher

	doTheNeedful: IngredientsFunction = astModuleToIngredientsFunction(astModule, sourceCallableIdentifier)
	doTheNeedful.imports.update(shatteredDataclass.imports)
	targetCallableIdentifier = ingredientsFunction.astFunctionDef.name
	doTheNeedful = unpackDataclassCallFunctionRepackDataclass(doTheNeedful, targetCallableIdentifier, shatteredDataclass)

	"""
	Put pack/unpack and concurrency is separate functions.
	- make the master copy of the state
	- implement concurrency
	- change the dataclass from MapFoldingState to ParallelMapFoldingState
		- import
		- parameter annotation
		- return annotation
		- when repacking, call the new identifier
	- unpack the new fields
	- have a master groupsOfFolds with the total from all taskIndex
	- repack the new fields
	- store each taskIndex state in a yet-to-be-created array (probably a primitive list indexed by taskIndex): this is strictly for the research purposes. Someone may want to examine and compare the states of the task divisions.
	"""

	# THIS is fanfreakingtastic. This dynamically updates the calling function (doTheNeedful) to have the correct
	# parameters in the call (to targetCallableIdentifier) and the correct identifiers in the return. NOTE that this
	# version of the function assumes that 1) the callee returns all of the parameters passed as arguments, 2) that the
	# callee parameters are in the exact same order as the return, and 3) that the identifiers are all exactly the same.
	astTuple: ast.Tuple | None = NodeTourist(Be.Return, Then.extractIt(DOT.value)).captureLastMatch(ingredientsFunction.astFunctionDef)
	if astTuple is None: raise raiseIfNoneGitHubIssueNumber3
	astTuple.ctx = ast.Store()
	findThis = ClassIsAndAttribute.valueIs(ast.Assign, IfThis.isCall_Identifier(targetCallableIdentifier))
	doThat = Then.replaceWith(Make.Assign([astTuple], value=Make.Call(Make.Name(targetCallableIdentifier), astTuple.elts)))
	changeAssignCallToTarget = NodeChanger(findThis, doThat)
	changeAssignCallToTarget.visit(doTheNeedful.astFunctionDef)

	ingredientsModule = IngredientsModule([ingredientsFunction, doTheNeedful])
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
	moduleIdentifierHARDCODED: ast_Identifier = 'dataPacking'
	callableIdentifierHARDCODED: ast_Identifier = 'sequential'

	algorithmSourceModule = algorithmSourceModuleHARDCODED
	sourceCallableIdentifier = The.sourceCallableDispatcher
	logicalPathSourceModule = '.'.join([The.packageName, algorithmSourceModule])

	logicalPathInfix = logicalPathInfixHARDCODED
	moduleIdentifier = moduleIdentifierHARDCODED
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
	astTuple: ast.Tuple | None = NodeTourist(Be.Return, Then.extractIt(DOT.value)).captureLastMatch(targetFunctionDef)
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
	makeDaoOfMapFoldingParallelV1()
