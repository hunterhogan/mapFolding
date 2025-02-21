"""I think this module is free of hardcoded values.
TODO: consolidate the logic in this module."""
from mapFolding.someAssemblyRequired.synthesizeNumbaGeneralized import *

def insertArrayIn_body(FunctionDefTarget: ast.FunctionDef, identifier: str, arrayTarget: numpy.ndarray, allImports: UniversalImportTracker, unrollSlices: int | None = None) -> tuple[ast.FunctionDef, UniversalImportTracker]:
	arrayType = type(arrayTarget)
	moduleConstructor = arrayType.__module__
	constructorName = arrayType.__name__
	# NOTE hack
	constructorName = constructorName.replace('ndarray', 'array')
	argData_dtype: numpy.dtype = arrayTarget.dtype
	datatypeName = argData_dtype.name
	dtypeAsName = f"{moduleConstructor}_{datatypeName}"

	allImports.addImportFromStr(moduleConstructor, constructorName)
	allImports.addImportFromStr(moduleConstructor, datatypeName, dtypeAsName)

	def insertAssign(assignee: str, arraySlice: numpy.ndarray) -> None:
		nonlocal FunctionDefTarget
		onlyDataRLE = autoDecodingRLE(arraySlice, addSpaces=True)
		astStatement = cast(ast.Expr, ast.parse(onlyDataRLE).body[0])
		dataAst = astStatement.value

		arrayCall = Then.make_astCall(name=constructorName, args=[dataAst], list_astKeywords=[ast.keyword(arg='dtype', value=ast.Name(id=dtypeAsName, ctx=ast.Load()))])

		assignment = ast.Assign(targets=[ast.Name(id=assignee, ctx=ast.Store())], value=arrayCall)#NOTE
		FunctionDefTarget.body.insert(0, assignment)

	if not unrollSlices:
		insertAssign(identifier, arrayTarget)
	else:
		for index, arraySlice in enumerate(arrayTarget):
			insertAssign(f"{identifier}_{index}", arraySlice)

	return FunctionDefTarget, allImports

def findAndReplaceTrackArrayIn_body(FunctionDefTarget: ast.FunctionDef, identifier: str , arrayTarget: numpy.ndarray , allImports: UniversalImportTracker) -> tuple[ast.FunctionDef, UniversalImportTracker]:

	arrayType = type(arrayTarget)
	moduleConstructor = arrayType.__module__
	constructorName = arrayType.__name__
	# NOTE hack
	constructorName = constructorName.replace('ndarray', 'array')
	allImports.addImportFromStr(moduleConstructor, constructorName)

	for statement in FunctionDefTarget.body.copy():
		if ifThis.isUnpackingAnArray(identifier)(statement):
			datatypeName = hackSSOTdatatype(statement.targets[0].id) # type: ignore
			dtypeAsName = f"{moduleConstructor}_{datatypeName}"
			indexAsStr = ast.unparse(statement.value.slice) # type: ignore
			arraySlice = arrayTarget[eval(indexAsStr)]

			onlyDataRLE = autoDecodingRLE(arraySlice, addSpaces=True)
			astStatement = cast(ast.Expr, ast.parse(onlyDataRLE).body[0])
			dataAst = astStatement.value

			arrayCall = Then.make_astCall(name=constructorName, args=[dataAst], list_astKeywords=[ast.keyword(arg='dtype', value=ast.Name(id=dtypeAsName, ctx=ast.Load()))])

			assignment = ast.Assign(targets=[statement.targets[0]], value=arrayCall) # type: ignore
			FunctionDefTarget.body.insert(0, assignment)
			FunctionDefTarget.body.remove(statement)
			allImports.addImportFromStr(moduleConstructor, datatypeName, dtypeAsName)
	return FunctionDefTarget, allImports

def findAndReplaceArraySubscriptIn_body(FunctionDefTarget: ast.FunctionDef, identifier: str, arrayTarget: numpy.ndarray, Z0Z_listChaff: list[str], allImports: UniversalImportTracker) -> tuple[ast.FunctionDef, UniversalImportTracker]:
	moduleConstructor = Z0Z_getDatatypeModuleScalar()
	for statement in FunctionDefTarget.body.copy():
		if ifThis.isUnpackingAnArray(identifier)(statement):
			astSubscript: ast.Subscript = statement.value # type: ignore
			astAssignee: ast.Name = statement.targets[0] # type: ignore
			argData_dtypeName = hackSSOTdatatype(astAssignee.id)
			allImports.addImportFromStr(moduleConstructor, argData_dtypeName)
			indexAs_astAttribute: ast.Attribute = astSubscript.slice # type: ignore
			indexAsStr = ast.unparse(indexAs_astAttribute)
			argDataSlice: int = arrayTarget[eval(indexAsStr)].item()
			astCall = ast.Call(func=ast.Name(id=argData_dtypeName, ctx=ast.Load()), args=[ast.Constant(value=argDataSlice)], keywords=[])
			assignment = ast.Assign(targets=[astAssignee], value=astCall)
			if astAssignee.id not in Z0Z_listChaff:
				FunctionDefTarget.body.insert(0, assignment)
			FunctionDefTarget.body.remove(statement)
	return FunctionDefTarget, allImports

def removeAssignTargetFrom_body(FunctionDefTarget: ast.FunctionDef, identifier: str) -> ast.FunctionDef:
	# Remove assignment nodes where the target is either a Subscript referencing `identifier` or satisfies ifThis.nameIs(identifier).
	def predicate(astNode: ast.AST) -> bool:
		if not isinstance(astNode, ast.Assign) or not astNode.targets:
			return False
		targetNode = astNode.targets[0]
		return (isinstance(targetNode, ast.Subscript) and isinstance(targetNode.value, ast.Name) and targetNode.value.id == identifier) or ifThis.nameIs(identifier)(targetNode)
	def replacementBuilder(astNode: ast.AST) -> ast.stmt | None:
		# Returning None removes the node.
		return None
	FunctionDefSherpa = NodeReplacer(predicate, replacementBuilder).visit(FunctionDefTarget)
	if not FunctionDefSherpa:
		raise FREAKOUT("Dude, where's my function?")
	else:
		FunctionDefTarget = cast(ast.FunctionDef, FunctionDefSherpa)
	ast.fix_missing_locations(FunctionDefTarget)
	return FunctionDefTarget

def findAndReplaceAnnAssignIn_body(FunctionDefTarget: ast.FunctionDef, allImports: UniversalImportTracker) -> tuple[ast.FunctionDef, UniversalImportTracker]:
	moduleConstructor = Z0Z_getDatatypeModuleScalar()
	for stmt in FunctionDefTarget.body.copy():
		if isinstance(stmt, ast.AnnAssign):
			if isinstance(stmt.target, ast.Name) and isinstance(stmt.value, ast.Constant):
				astAssignee: ast.Name = stmt.target
				argData_dtypeName = hackSSOTdatatype(astAssignee.id)
				allImports.addImportFromStr(moduleConstructor, argData_dtypeName)
				astCall = ast.Call(func=ast.Name(id=argData_dtypeName, ctx=ast.Load()) , args=[stmt.value], keywords=[])
				assignment = ast.Assign(targets=[astAssignee], value=astCall)
				FunctionDefTarget.body.insert(0, assignment)
				FunctionDefTarget.body.remove(stmt)
	return FunctionDefTarget, allImports

def findThingyReplaceWithConstantIn_body(FunctionDefTarget: ast.FunctionDef, object: str, value: int) -> ast.FunctionDef:
	"""
	Replaces nodes in astFunction matching the AST of the string `object`
	with a constant node holding the provided value.
	"""
	targetExpression = ast.parse(object, mode='eval').body
	targetDump = ast.dump(targetExpression, annotate_fields=False)

	def findNode(node: ast.AST) -> bool:
		return ast.dump(node, annotate_fields=False) == targetDump

	def replaceWithConstant(node: ast.AST) -> ast.AST:
		return ast.copy_location(ast.Constant(value=value), node)

	transformer = NodeReplacer(findNode, replaceWithConstant)
	newFunction = cast(ast.FunctionDef, transformer.visit(FunctionDefTarget))
	ast.fix_missing_locations(newFunction)
	return newFunction

def findAstNameReplaceWithConstantIn_body(FunctionDefTarget: ast.FunctionDef, name: str, value: int) -> ast.FunctionDef:
	def replaceWithConstant(node: ast.AST) -> ast.AST:
		return ast.copy_location(ast.Constant(value=value), node)

	return cast(ast.FunctionDef, NodeReplacer(ifThis.nameIs(name), replaceWithConstant).visit(FunctionDefTarget))

def insertReturnStatementIn_body(FunctionDefTarget: ast.FunctionDef, arrayTarget: numpy.ndarray, allImports: UniversalImportTracker) -> tuple[ast.FunctionDef, UniversalImportTracker]:
	"""Add multiplication and return statement to function, properly constructing AST nodes."""
	# Create AST for multiplication operation
	multiplicand = Z0Z_identifierCountFolds
	datatype = hackSSOTdatatype(multiplicand)
	multiplyOperation = ast.BinOp(
		left=ast.Name(id=multiplicand, ctx=ast.Load()),
		op=ast.Mult(), right=ast.Constant(value=int(arrayTarget[-1])))

	returnStatement = ast.Return(value=multiplyOperation)

	datatype = hackSSOTdatatype(Z0Z_identifierCountFolds)
	FunctionDefTarget.returns = ast.Name(id=datatype, ctx=ast.Load())
	datatypeModuleScalar = Z0Z_getDatatypeModuleScalar()
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
			self.iteratorName = iteratorName
			self.constantValue = constantValue

		def visit_Name(self, node: ast.Name) -> ast.AST:
			if node.id == self.iteratorName:
				return ast.copy_location(ast.Constant(value=self.constantValue), node)
			return self.generic_visit(node)

	# NodeTransformer that finds while loops (even if deeply nested) and unrolls them.
	class WhileLoopUnroller(ast.NodeTransformer):
		def __init__(self, iteratorName: str, iterationsTotal: int) -> None:
			super().__init__()
			self.iteratorName = iteratorName
			self.iterationsTotal = iterationsTotal

		def visit_While(self, node: ast.While) -> list[ast.stmt]:
				# Check if the while loop's test uses the iterator.
			if isinstance(node.test, ast.Compare) and ifThis.nameIs(self.iteratorName)(node.test.left):
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
						copiedStatement = copy.deepcopy(loopStatement)
						replacer = ReplaceIterator(self.iteratorName, iterationIndex)
						newStatement = replacer.visit(copiedStatement)
						ast.fix_missing_locations(newStatement)
						newStatements.append(newStatement)
				# Optionally, process the orelse block.
				if node.orelse:
					for elseStmt in node.orelse:
						visitedElse = self.visit(elseStmt)
						if isinstance(visitedElse, list):
							newStatements.extend(visitedElse)
						else:
							newStatements.append(visitedElse)
				return newStatements
			return [cast(ast.stmt, self.generic_visit(node))]

	newFunctionDef = WhileLoopUnroller(iteratorName, iterationsTotal).visit(FunctionDefTarget)
	ast.fix_missing_locations(newFunctionDef)
	return newFunctionDef

def makeLauncherBasicJobNumba(callableTarget: str, pathFilenameFoldsTotal: Path) -> ast.Module:
	linesLaunch = f"""
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

def makeFunctionDef(astModule: ast.Module,
					callableTarget: str,
					parametersNumba: ParametersNumba | None = None,
					inlineCallables: bool | None = False,
					unpackArrays: bool | None = False,
					allImports: UniversalImportTracker | None = None) -> tuple[ast.FunctionDef, UniversalImportTracker]:
	if allImports is None:
		allImports = UniversalImportTracker()
	for statement in astModule.body:
		if isinstance(statement, (ast.Import, ast.ImportFrom)):
			allImports.addAst(statement)

	if inlineCallables:
		dictionaryFunctionDef = {statement.name: statement for statement in astModule.body if isinstance(statement, ast.FunctionDef)}
		callableInlinerWorkhorse = RecursiveInliner(dictionaryFunctionDef)
		# NOTE the inliner assumes each function is not called more than once
		# TODO change the inliner to handle multiple calls to the same function
		FunctionDefTarget = callableInlinerWorkhorse.inlineFunctionBody(callableTarget)
	else:
		FunctionDefTarget = next((node for node in astModule.body if isinstance(node, ast.FunctionDef) and node.name == callableTarget), None)
	if not FunctionDefTarget:
		raise ValueError(f"Could not find function {callableTarget} in source code")

	ast.fix_missing_locations(FunctionDefTarget)

	FunctionDefTarget, allImports = decorateCallableWithNumba(FunctionDefTarget, allImports, parametersNumba)

	# NOTE vestigial hardcoding
	if unpackArrays:
		for tupleUnpack in [(indexMy, 'my'), (indexTrack, 'track')]:
			unpacker = UnpackArrays(*tupleUnpack)
			FunctionDefTarget = cast(ast.FunctionDef, unpacker.visit(FunctionDefTarget))
			ast.fix_missing_locations(FunctionDefTarget)

	return FunctionDefTarget, allImports

def decorateCallableWithNumba(FunctionDefTarget: ast.FunctionDef, allImports: UniversalImportTracker, parametersNumba: ParametersNumba | None = None) -> tuple[ast.FunctionDef, UniversalImportTracker]:
	def Z0Z_UnhandledDecorators(astCallable: ast.FunctionDef) -> ast.FunctionDef:
		# TODO: more explicit handling of decorators. I'm able to ignore this because I know `algorithmSource` doesn't have any decorators.
		for decoratorItem in astCallable.decorator_list.copy():
			import warnings
			astCallable.decorator_list.remove(decoratorItem)
			warnings.warn(f"Removed decorator {ast.unparse(decoratorItem)} from {astCallable.name}")
		return astCallable

	def make_arg4parameter(signatureElement: ast.arg) -> ast.Subscript | None:
		if isinstance(signatureElement.annotation, ast.Subscript) and isinstance(signatureElement.annotation.slice, ast.Tuple):
			annotationShape = signatureElement.annotation.slice.elts[0]
			if isinstance(annotationShape, ast.Subscript) and isinstance(annotationShape.slice, ast.Tuple):
				shapeAsListSlices = [ast.Slice() for axis in range(len(annotationShape.slice.elts))]
				shapeAsListSlices[-1] = ast.Slice(step=ast.Constant(value=1))
				shapeAST = ast.Tuple(elts=list(shapeAsListSlices), ctx=ast.Load())
			else:
				shapeAST = ast.Slice(step=ast.Constant(value=1))

			annotationDtype = signatureElement.annotation.slice.elts[1]
			if (isinstance(annotationDtype, ast.Subscript) and isinstance(annotationDtype.slice, ast.Attribute)):
				datatypeAST = annotationDtype.slice.attr
			else:
				datatypeAST = None

			ndarrayName = signatureElement.arg
			Z0Z_hacky_dtype = hackSSOTdatatype(ndarrayName)
			datatype_attr = datatypeAST or Z0Z_hacky_dtype
			allImports.addImportFromStr(datatypeModuleDecorator, datatype_attr)
			datatypeNumba = ast.Name(id=datatype_attr, ctx=ast.Load())

			return ast.Subscript(value=datatypeNumba, slice=shapeAST, ctx=ast.Load())
		return

	datatypeModuleDecorator = Z0Z_getDatatypeModuleScalar()
	list_argsDecorator: Sequence[ast.expr] = []

	list_arg4signature_or_function: list[ast.expr] = []
	for parameter in FunctionDefTarget.args.args:
		signatureElement = make_arg4parameter(parameter)
		if signatureElement:
			list_arg4signature_or_function.append(signatureElement)

	if FunctionDefTarget.returns and isinstance(FunctionDefTarget.returns, ast.Name):
		theReturn: ast.Name = FunctionDefTarget.returns
		list_argsDecorator = [cast(ast.expr, ast.Call(func=ast.Name(id=theReturn.id, ctx=ast.Load())
							, args=list_arg4signature_or_function if list_arg4signature_or_function else [] , keywords=[] ) )]
	elif list_arg4signature_or_function:
		list_argsDecorator = [cast(ast.expr, ast.Tuple(elts=list_arg4signature_or_function, ctx=ast.Load()))]

	for decorator in FunctionDefTarget.decorator_list.copy():
		if thisIsAnyNumbaJitDecorator(decorator):
			decorator = cast(ast.Call, decorator)
			if parametersNumba is None:
				parametersNumbaSherpa = Then.copy_astCallKeywords(decorator)
				if (HunterIsSureThereAreBetterWaysToDoThis := True):
					if parametersNumbaSherpa:
						parametersNumba = cast(ParametersNumba, parametersNumbaSherpa)
		FunctionDefTarget.decorator_list.remove(decorator)

	FunctionDefTarget = Z0Z_UnhandledDecorators(FunctionDefTarget)
	if parametersNumba is None:
		parametersNumba = parametersNumbaDEFAULT
	listDecoratorKeywords = [ast.keyword(arg=parameterName, value=ast.Constant(value=parameterValue)) for parameterName, parameterValue in parametersNumba.items()]

	decoratorModule = Z0Z_getDatatypeModuleScalar()
	decoratorCallable = Z0Z_getDecoratorCallable()
	allImports.addImportFromStr(decoratorModule, decoratorCallable)
	astDecorator = Then.make_astCall(decoratorCallable, list_argsDecorator, listDecoratorKeywords, None)

	FunctionDefTarget.decorator_list = [astDecorator]
	return FunctionDefTarget, allImports
