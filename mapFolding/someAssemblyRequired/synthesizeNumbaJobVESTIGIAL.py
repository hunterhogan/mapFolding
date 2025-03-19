"""Synthesize one file to compute `foldsTotal` of `mapShape`."""
from collections.abc import Sequence
from typing import Any, cast, TYPE_CHECKING
from mapFolding.filesystem import getFilenameFoldsTotal, getPathFilenameFoldsTotal
from mapFolding.someAssemblyRequired import ( ifThis, LedgerOfImports, Make, NodeReplacer, Then, )
from mapFolding.theSSOT import ( ComputationState, raiseIfNoneGitHubIssueNumber3, getPathJobRootDEFAULT, )
from os import PathLike
from pathlib import Path
from types import ModuleType
from Z0Z_tools import autoDecodingRLE
import ast
import python_minifier
import autoflake
import copy
import inspect
import numpy
if TYPE_CHECKING:
	from mapFolding.someAssemblyRequired.transformDataStructures import makeStateJobOUTDATED
	from mapFolding.someAssemblyRequired.ingredientsNumba import thisIsNumbaDotJit, decorateCallableWithNumba
	from mapFolding.someAssemblyRequired.ingredientsNumba import ParametersNumba, parametersNumbaDEFAULT

def Z0Z_gamma(FunctionDefTarget: ast.FunctionDef, astAssignee: ast.Name, statement: ast.Assign | ast.stmt, identifier: str, arrayTarget: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.integer[Any]]], allImports: LedgerOfImports) -> tuple[ast.FunctionDef, LedgerOfImports]:
	arrayType = type(arrayTarget)
	moduleConstructor: str = arrayType.__module__
	constructorName: str = arrayType.__name__.replace('ndarray', 'array') # NOTE hack
	dataAsStrRLE: str = autoDecodingRLE(arrayTarget, addSpaces=True)
	dataAs_astExpr: ast.expr = cast(ast.Expr, ast.parse(dataAsStrRLE).body[0]).value
	dtypeName: str = identifier
	dtypeAsName: str = f"{moduleConstructor}_{dtypeName}"
	list_astKeywords: list[ast.keyword] = [ast.keyword(arg='dtype', value=ast.Name(id=dtypeAsName, ctx=ast.Load()))]
	allImports.addImportFromStr(moduleConstructor, dtypeName, dtypeAsName)
	astCall: ast.Call = Make.astCall(Make.astName(constructorName), [dataAs_astExpr], list_astKeywords)
	assignment = ast.Assign(targets=[astAssignee], value=astCall)
	FunctionDefTarget.body.insert(0, assignment)
	FunctionDefTarget.body.remove(statement)
	allImports.addImportFromStr(moduleConstructor, constructorName)
	return FunctionDefTarget, allImports

def insertArrayIn_body(FunctionDefTarget: ast.FunctionDef, identifier: str, arrayTarget: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.integer[Any]]], allImports: LedgerOfImports, unrollSlices: int | None = None) -> tuple[ast.FunctionDef, LedgerOfImports]:
	def insertAssign(FunctionDefTarget: ast.FunctionDef, assignee: str, arraySlice: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.integer[Any]]], allImports: LedgerOfImports) -> tuple[ast.FunctionDef, LedgerOfImports]:
		statement = ast.Assign(targets=[ast.Name(id='beans', ctx=ast.Load())], value=ast.Constant(value='and cornbread'))
		FunctionDefTarget.body.insert(0, statement)
		astAssignee = ast.Name(id=assignee, ctx=ast.Store())
		return Z0Z_gamma(FunctionDefTarget, astAssignee, statement, identifier, arraySlice, allImports)

	if not unrollSlices:
		FunctionDefTarget, allImports = insertAssign(FunctionDefTarget, identifier, arrayTarget, allImports)
	else:
		for index, arraySlice in enumerate(arrayTarget):
			FunctionDefTarget, allImports = insertAssign(FunctionDefTarget, f"{identifier}_{index}", arraySlice, allImports)

	return FunctionDefTarget, allImports

def findAndReplaceTrackArrayIn_body(FunctionDefTarget: ast.FunctionDef, identifier: str, arrayTarget: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.integer[Any]]], allImports: LedgerOfImports) -> tuple[ast.FunctionDef, LedgerOfImports]:
	for statement in FunctionDefTarget.body.copy():
		if True:
			indexAsStr: str = ast.unparse(statement.value.slice) # type: ignore
			arraySlice: numpy.ndarray[Any, numpy.dtype[numpy.integer[Any]]] = arrayTarget[eval(indexAsStr)]
			astAssignee: ast.Name = cast(ast.Name, statement.targets[0]) # type: ignore
			FunctionDefTarget, allImports = Z0Z_gamma(FunctionDefTarget, astAssignee, statement, identifier, arraySlice, allImports)
	return FunctionDefTarget, allImports

def findAndReplaceArraySubscriptIn_body(FunctionDefTarget: ast.FunctionDef, identifier: str, arrayTarget: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.integer[Any]]], allImports: LedgerOfImports) -> tuple[ast.FunctionDef, LedgerOfImports]:
	# parameter: I define moduleConstructor
	moduleConstructor = 'numba'

	for statement in FunctionDefTarget.body.copy():
		if True:
			indexAsStr: str = ast.unparse(statement.value.slice) # type: ignore
			arraySlice: numpy.ndarray[Any, numpy.dtype[numpy.integer[Any]]] = arrayTarget[eval(indexAsStr)]
			astAssignee: ast.Name = cast(ast.Name, statement.targets[0]) # type: ignore
			arraySliceItem: int = arraySlice.item()
			constructorName: str = astAssignee.id
			dataAs_astExpr = ast.Constant(value=arraySliceItem)
			list_astKeywords: list[ast.keyword] = []
			astCall: ast.Call = Make.astCall(Make.astName(constructorName), [dataAs_astExpr], list_astKeywords)
			assignment = ast.Assign(targets=[astAssignee], value=astCall)
			FunctionDefTarget.body.insert(0, assignment)
			FunctionDefTarget.body.remove(statement)
			allImports.addImportFromStr(moduleConstructor, constructorName)
	return FunctionDefTarget, allImports

def removeAssignmentFrom_body(FunctionDefTarget: ast.FunctionDef, identifier: str) -> ast.FunctionDef:
	FunctionDefSherpa: ast.AST | Sequence[ast.AST] | None = NodeReplacer(ifThis.isAnyAssignmentTo(identifier), Then.removeThis).visit(FunctionDefTarget)
	if not FunctionDefSherpa:
		raise raiseIfNoneGitHubIssueNumber3("Dude, where's my function?")
	else:
		FunctionDefTarget = cast(ast.FunctionDef, FunctionDefSherpa)
	ast.fix_missing_locations(FunctionDefTarget)
	return FunctionDefTarget

def findAndReplaceAnnAssignIn_body(FunctionDefTarget: ast.FunctionDef, allImports: LedgerOfImports) -> tuple[ast.FunctionDef, LedgerOfImports]:
	"""Unlike most of the other functions, this is generic: it tries to turn an annotation into a construction call."""
	moduleConstructor: str = 'numba'
	for stmt in FunctionDefTarget.body.copy():
		if isinstance(stmt, ast.AnnAssign):
			if isinstance(stmt.target, ast.Name) and isinstance(stmt.value, ast.Constant):
				astAssignee: ast.Name = stmt.target
				argData_dtypeName: str = astAssignee.id
				allImports.addImportFromStr(moduleConstructor, argData_dtypeName)
				astCall = ast.Call(func=ast.Name(id=argData_dtypeName, ctx=ast.Load()), args=[stmt.value], keywords=[])
				assignment = ast.Assign(targets=[astAssignee], value=astCall)
				FunctionDefTarget.body.insert(0, assignment)
				FunctionDefTarget.body.remove(stmt)
	return FunctionDefTarget, allImports

def findThingyReplaceWithConstantIn_body(FunctionDefTarget: ast.FunctionDef, object: str, value: int) -> ast.FunctionDef:
	"""
	Replaces nodes in astFunction matching the AST of the string `object`
	with a constant node holding the provided value.
	"""
	targetExpression: ast.expr = ast.parse(object, mode='eval').body
	targetDump: str = ast.dump(targetExpression, annotate_fields=False)

	def findNode(node: ast.AST) -> bool:
		return ast.dump(node, annotate_fields=False) == targetDump

	def replaceWithConstant(node: ast.AST) -> ast.AST:
		return ast.copy_location(ast.Constant(value=value), node)

	transformer = NodeReplacer(findNode, replaceWithConstant)
	newFunction: ast.FunctionDef = cast(ast.FunctionDef, transformer.visit(FunctionDefTarget))
	ast.fix_missing_locations(newFunction)
	return newFunction

def findAstNameReplaceWithConstantIn_body(FunctionDefTarget: ast.FunctionDef, name: str, value: int) -> ast.FunctionDef:
	def replaceWithConstant(node: ast.AST) -> ast.AST:
		return ast.copy_location(ast.Constant(value=value), node)

	return cast(ast.FunctionDef, NodeReplacer(ifThis.isName_Identifier(name), replaceWithConstant).visit(FunctionDefTarget))

def insertReturnStatementIn_body(FunctionDefTarget: ast.FunctionDef, arrayTarget: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.integer[Any]]], allImports: LedgerOfImports) -> tuple[ast.FunctionDef, LedgerOfImports]:
	"""Add multiplication and return statement to function, properly constructing AST nodes."""
	# Create AST for multiplication operation
	multiplicand = 'Z0Z_identifierCountFolds'
	multiplyOperation = ast.BinOp(
		left=ast.Name(id=multiplicand, ctx=ast.Load()),
		op=ast.Mult(), right=ast.Constant(value=int(arrayTarget[-1])))

	returnStatement = ast.Return(value=multiplyOperation)

	datatype: str = 'Z0Z_identifierCountFolds'
	FunctionDefTarget.returns = ast.Name(id=datatype, ctx=ast.Load())
	datatypeModuleScalar: str = 'numba'
	allImports.addImportFromStr(datatypeModuleScalar, datatype)

	FunctionDefTarget.body.append(returnStatement)

	return FunctionDefTarget, allImports

def findAndReplaceWhileLoopIn_body(FunctionDefTarget: ast.FunctionDef, iteratorName: str, iterationsTotal: int) -> ast.FunctionDef:
	"""
	Unroll all nested while loops matching the condition that their test uses `iteratorName`.
	"""
	# Helper transformer to replace iterator occurrences with a constant.
	class ReplaceIterator(ast.NodeTransformer):
		def __init__(self, iteratorName: str, constantValue: int) -> None:
			super().__init__()
			self.iteratorName: str = iteratorName
			self.constantValue: int = constantValue

		def visit_Name(self, node: ast.Name) -> ast.AST:
			if node.id == self.iteratorName:
				return ast.copy_location(ast.Constant(value=self.constantValue), node)
			return self.generic_visit(node)

	# NodeTransformer that finds while loops (even if deeply nested) and unrolls them.
	class WhileLoopUnroller(ast.NodeTransformer):
		def __init__(self, iteratorName: str, iterationsTotal: int) -> None:
			super().__init__()
			self.iteratorName: str = iteratorName
			self.iterationsTotal: int = iterationsTotal

		def visit_While(self, node: ast.While) -> list[ast.stmt]:
				# Check if the while loop's test uses the iterator.
			if isinstance(node.test, ast.Compare) and ifThis.isName_Identifier(self.iteratorName)(node.test.left):
				# Recurse the while loop body and remove AugAssign that increments the iterator.
				cleanBodyStatements: list[ast.stmt] = []
				for loopStatement in node.body:
					# Recursively visit nested statements.
					visitedStatement = self.visit(loopStatement)
					# Remove direct AugAssign: iterator += 1.
					if (isinstance(loopStatement, ast.AugAssign) and
						isinstance(loopStatement.target, ast.Name) and
						loopStatement.target.id == self.iteratorName and
						isinstance(loopStatement.op, ast.Add) and
						isinstance(loopStatement.value, ast.Constant) and
						loopStatement.value.value == 1):
						continue
					cleanBodyStatements.append(visitedStatement)

				newStatements: list[ast.stmt] = []
				# Unroll using the filtered body.
				for iterationIndex in range(self.iterationsTotal):
					for loopStatement in cleanBodyStatements:
						copiedStatement: ast.stmt = copy.deepcopy(loopStatement)
						replacer = ReplaceIterator(self.iteratorName, iterationIndex)
						newStatement = replacer.visit(copiedStatement)
						ast.fix_missing_locations(newStatement)
						newStatements.append(newStatement)
				# Optionally, process the orelse block.
				if node.orelse:
					for elseStmt in node.orelse:
						visitedElse = self.visit(elseStmt)
						if isinstance(visitedElse, list):
							newStatements.extend(cast(list[ast.stmt], visitedElse))
						else:
							newStatements.append(visitedElse)
				return newStatements
			return [cast(ast.stmt, self.generic_visit(node))]

	newFunctionDef = WhileLoopUnroller(iteratorName, iterationsTotal).visit(FunctionDefTarget)
	ast.fix_missing_locations(newFunctionDef)
	return newFunctionDef

def makeLauncherTqdmJobNumba(callableTarget: str, pathFilenameFoldsTotal: Path, totalEstimated: int, leavesTotal:int) -> ast.Module:
	linesLaunch: str = f"""
if __name__ == '__main__':
	with ProgressBar(total={totalEstimated}, update_interval=2) as statusUpdate:
		{callableTarget}(statusUpdate)
		foldsTotal = statusUpdate.n * {leavesTotal}
		print("", foldsTotal)
		writeStream = open('{pathFilenameFoldsTotal.as_posix()}', 'w')
		writeStream.write(str(foldsTotal))
		writeStream.close()
"""
	return ast.parse(linesLaunch)

def makeLauncherBasicJobNumba(callableTarget: str, pathFilenameFoldsTotal: Path) -> ast.Module:
	linesLaunch: str = f"""
if __name__ == '__main__':
	import time
	timeStart = time.perf_counter()
	foldsTotal = {callableTarget}()
	print(foldsTotal, time.perf_counter() - timeStart)
	writeStream = open('{pathFilenameFoldsTotal.as_posix()}', 'w')
	writeStream.write(str(foldsTotal))
	writeStream.close()
"""
	return ast.parse(linesLaunch)

def doUnrollCountGaps(FunctionDefTarget: ast.FunctionDef, stateJob: ComputationState, allImports: LedgerOfImports) -> tuple[ast.FunctionDef, LedgerOfImports]:
	"""The initial results were very bad."""
	FunctionDefTarget = findAndReplaceWhileLoopIn_body(FunctionDefTarget, 'indexDimension', stateJob.dimensionsTotal)
	FunctionDefTarget = removeAssignmentFrom_body(FunctionDefTarget, 'indexDimension')
	FunctionDefTarget = removeAssignmentFrom_body(FunctionDefTarget, 'connectionGraph')
	FunctionDefTarget, allImports = insertArrayIn_body(FunctionDefTarget, 'connectionGraph', stateJob.connectionGraph, allImports, stateJob.dimensionsTotal)
	for index in range(stateJob.dimensionsTotal):
		class ReplaceConnectionGraph(ast.NodeTransformer):
			def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
				node = cast(ast.Subscript, self.generic_visit(node))
				if (isinstance(node.value, ast.Name) and node.value.id == "connectionGraph" and
					isinstance(node.slice, ast.Tuple) and len(node.slice.elts) >= 1):
					firstElement: ast.expr = node.slice.elts[0]
					if isinstance(firstElement, ast.Constant) and firstElement.value == index:
						newName = ast.Name(id=f"connectionGraph_{index}", ctx=ast.Load())
						remainingIndices: list[ast.expr] = node.slice.elts[1:]
						if len(remainingIndices) == 1:
							newSlice: ast.expr = remainingIndices[0]
						else:
							newSlice = ast.Tuple(elts=remainingIndices, ctx=ast.Load())
						return ast.copy_location(ast.Subscript(value=newName, slice=newSlice, ctx=node.ctx), node)
				return node
		transformer = ReplaceConnectionGraph()
		FunctionDefTarget = transformer.visit(FunctionDefTarget)
	return FunctionDefTarget, allImports

def writeJobNumba(mapShape: Sequence[int], algorithmSource: ModuleType, callableTarget: str | None = None, parametersNumba: ParametersNumba | None = None, pathFilenameWriteJob: str | PathLike[str] | None = None, unrollCountGaps: bool | None = False, Z0Z_totalEstimated: int = 0, **keywordArguments: Any | None) -> Path:
	""" Parameters: **keywordArguments: most especially for `computationDivisions` if you want to make a parallel job. Also `CPUlimit`.
	Notes:
	Hypothetically, everything can now be configured with parameters and functions. And changing how the job is written is relatively easy.

	Overview
	- the code starts life in theDao.py, which has many optimizations; `makeNumbaOptimizedFlow` increase optimization especially by using numba; `writeJobNumba` increases optimization especially by limiting its capabilities to just one set of parameters
	- the synthesized module must run well as a standalone interpreted-Python script
	- the next major optimization step will (probably) be to use the module synthesized by `writeJobNumba` to compile a standalone executable
	- Nevertheless, at each major optimization step, the code is constantly being improved and optimized, so everything must be well organized and able to handle upstream and downstream changes

	Minutia
	- perf_counter is for testing. When I run a real job, I delete those lines
	- avoid `with` statement

	Necessary
	- Move the function's parameters to the function body,
	- initialize identifiers with their state types and values,

	Optimizations
	- replace static-valued identifiers with their values
	- narrowly focused imports
	"""

	# NOTE get the raw ingredients: data and the algorithm
	stateJob = makeStateJobOUTDATED(mapShape, writeJob=False, **keywordArguments)
	pythonSource: str = inspect.getsource(algorithmSource)
	astModule: ast.Module = ast.parse(pythonSource)
	setFunctionDef: set[ast.FunctionDef] = {statement for statement in astModule.body if isinstance(statement, ast.FunctionDef)}

	if not callableTarget:
		if len(setFunctionDef) == 1:
			FunctionDefTarget = setFunctionDef.pop()
			callableTarget = FunctionDefTarget.name
		else:
			raise ValueError(f"I did not receive a `callableTarget` and {algorithmSource.__name__=} has more than one callable: {setFunctionDef}. Please select one.")
	else:
		listFunctionDefTarget: list[ast.FunctionDef] = [statement for statement in setFunctionDef if statement.name == callableTarget]
		FunctionDefTarget = listFunctionDefTarget[0] if listFunctionDefTarget else None # type: ignore
	if not FunctionDefTarget: raise ValueError(f"I received `{callableTarget=}` and {algorithmSource.__name__=}, but I could not find that function in that source.")

	# NOTE `allImports` is a complementary container to `FunctionDefTarget`; the `FunctionDefTarget` cannot track its own imports very well.
	allImports = LedgerOfImports(astModule)

	# NOTE remove the parameters from the function signature
	for pirateScowl in FunctionDefTarget.args.args.copy():
		match pirateScowl.arg:
			case 'connectionGraph':
				FunctionDefTarget, allImports = insertArrayIn_body(FunctionDefTarget, pirateScowl.arg, stateJob.connectionGraph, allImports)
			case 'gapsWhere':
				FunctionDefTarget, allImports = insertArrayIn_body(FunctionDefTarget, pirateScowl.arg, stateJob.gapsWhere, allImports)
			case 'foldGroups':
				FunctionDefTarget = removeAssignmentFrom_body(FunctionDefTarget, pirateScowl.arg)
			case _:
				pass
		FunctionDefTarget.args.args.remove(pirateScowl)

	identifierCounter = 'Z0Z_identifierCountFolds'
	astExprIncrementCounter = ast.Expr(value = Make.astCall(Make.nameDOTname(identifierCounter, 'update'), listArguments=[ast.Constant(value=1)], list_astKeywords=[]))
	FunctionDefTarget= cast(ast.FunctionDef, NodeReplacer(ifThis.isAugAssignTo(identifierCounter), Then.replaceWith(astExprIncrementCounter)).visit(FunctionDefTarget))
	ast.fix_missing_locations(FunctionDefTarget)

	for assignmentTarget in ['taskIndex', 'dimensionsTotal', identifierCounter]:
		FunctionDefTarget = removeAssignmentFrom_body(FunctionDefTarget, assignmentTarget)
	# NOTE replace identifiers with static values with their values
	FunctionDefTarget = findAstNameReplaceWithConstantIn_body(FunctionDefTarget, 'dimensionsTotal', int(stateJob.dimensionsTotal))
	FunctionDefTarget = findThingyReplaceWithConstantIn_body(FunctionDefTarget, 'foldGroups[-1]', int(stateJob.foldGroups[-1]))

	# NOTE an attempt at optimization
	if unrollCountGaps:
		FunctionDefTarget, allImports = doUnrollCountGaps(FunctionDefTarget, stateJob, allImports)

	# NOTE starting the count and printing the total
	pathFilenameFoldsTotal: Path = getPathFilenameFoldsTotal(stateJob.mapShape)

	astLauncher: ast.Module = makeLauncherBasicJobNumba(FunctionDefTarget.name, pathFilenameFoldsTotal)

	# TODO create function for assigning value to `totalEstimated`
	totalEstimated: int = Z0Z_totalEstimated
	astLauncher = makeLauncherTqdmJobNumba(FunctionDefTarget.name, pathFilenameFoldsTotal, totalEstimated, stateJob.foldGroups[-1])

	allImports.addImportFromStr('numba_progress', 'ProgressBar')
	allImports.addImportFromStr('numba_progress', 'ProgressBarType')

	# add ProgressBarType parameter to function args
	counterArg = ast.arg(arg=identifierCounter, annotation=ast.Name(id='ProgressBarType', ctx=ast.Load()))
	FunctionDefTarget.args.args.append(counterArg)

	if parametersNumba is None:
		parametersNumba = parametersNumbaDEFAULT
	parametersNumba['nogil'] = True

	FunctionDefTarget, allImports = insertReturnStatementIn_body(FunctionDefTarget, stateJob.foldGroups, allImports)

	FunctionDefTarget, allImports = findAndReplaceAnnAssignIn_body(FunctionDefTarget, allImports)
	# NOTE add the perfect decorator
	if thisIsNumbaDotJit(FunctionDefTarget.decorator_list[0]):
		astCall: ast.Call = cast(ast.Call, FunctionDefTarget.decorator_list[0])
		astCall.func = ast.Name(id='jit', ctx=ast.Load())
		FunctionDefTarget.decorator_list[0] = astCall

	# NOTE add imports, make str, remove unused imports
	astImports: list[ast.ImportFrom | ast.Import] = allImports.makeListAst()
	astModule = ast.Module(body=cast(list[ast.stmt], astImports + [FunctionDefTarget] + [astLauncher]), type_ignores=[])
	ast.fix_missing_locations(astModule)
	pythonSource = ast.unparse(astModule)
	pythonSource = autoflake.fix_code(pythonSource, ['mapFolding', 'numba', 'numpy'])
	pythonSource = python_minifier.minify(pythonSource, remove_annotations = False, remove_pass = False, remove_literal_statements = False, combine_imports = True, hoist_literals = False, rename_locals = False, rename_globals = False, remove_object_base = False, convert_posargs_to_args = False, preserve_shebang = True, remove_asserts = False, remove_debug = False, remove_explicit_return_none = False, remove_builtin_exception_brackets = False, constant_folding = False)

	# NOTE put on disk
	if pathFilenameWriteJob is None:
		filename: str = getFilenameFoldsTotal(stateJob.mapShape)
		pathRoot: Path = getPathJobRootDEFAULT()
		pathFilenameWriteJob = Path(pathRoot, Path(filename).stem, Path(filename).with_suffix('.py'))
	else:
		pathFilenameWriteJob = Path(pathFilenameWriteJob)
	pathFilenameWriteJob.parent.mkdir(parents=True, exist_ok=True)

	pathFilenameWriteJob.write_text(pythonSource)

	return pathFilenameWriteJob

if __name__ == '__main__':
	mapShape: list[int] = [5,5]
	dictionaryEstimates: dict[tuple[int, ...], int] = {
		(2,2,2,2,2,2,2,2): 362794844160000,
		(2,21): 1493028892051200,
		(3,15): 9842024675968800,
		(3,3,3,3,3): 85109616000000,
		(3,3,3,3): 85109616000,
		(8,8): 129950723279272000,
	}

	totalEstimated: int = dictionaryEstimates.get(tuple(mapShape), 10**8)
	from mapFolding.syntheticModules import numbaCount_doTheNeedful
	algorithmSource: ModuleType = numbaCount_doTheNeedful

	callableTarget = 'countSequential'

	parametersNumba: ParametersNumba = parametersNumbaDEFAULT
	parametersNumba['nogil'] = True
	parametersNumba['boundscheck'] = False

	pathFilenameWriteJob = None

	writeJobNumba(mapShape, algorithmSource, callableTarget, parametersNumba, pathFilenameWriteJob, Z0Z_totalEstimated=totalEstimated)
