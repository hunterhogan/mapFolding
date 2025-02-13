"""I think this module is free of hardcoded values.
TODO: consolidate the logic in this module."""
from mapFolding.someAssemblyRequired.synthesizeNumbaGeneralized import *

def Z0Z_OneCallable(pythonSource: str, callableTarget: str, parametersNumba: Optional[ParametersNumba]=None, inlineCallables: Optional[bool]=False , unpackArrays: Optional[bool]=False , allImports: Optional[UniversalImportTracker]=None ) -> str:
	astModule: ast.Module = ast.parse(pythonSource, type_comments=True)
	if allImports is None:
		allImports = UniversalImportTracker()

	for statement in astModule.body:
		if isinstance(statement, (ast.Import, ast.ImportFrom)):
			allImports.addAst(statement)

	if inlineCallables:
		dictionaryFunctionDef = {statement.name: statement for statement in astModule.body if isinstance(statement, ast.FunctionDef)}
		callableInlinerWorkhorse = RecursiveInliner(dictionaryFunctionDef)
		FunctionDefTarget = callableInlinerWorkhorse.inlineFunctionBody(callableTarget)
	else:
		FunctionDefTarget = next((node for node in astModule.body if isinstance(node, ast.FunctionDef) and node.name == callableTarget), None)

	if not FunctionDefTarget:
		raise ValueError(f"Could not find function {callableTarget} in source code")

	ast.fix_missing_locations(FunctionDefTarget)

	FunctionDefTarget, allImports = decorateCallableWithNumba(FunctionDefTarget, allImports, parametersNumba)

	if unpackArrays:
		for tupleUnpack in [(indexMy, 'my'), (indexTrack, 'track')]:
			unpacker = UnpackArrays(*tupleUnpack)
			FunctionDefTarget = cast(ast.FunctionDef, unpacker.visit(FunctionDefTarget))
			ast.fix_missing_locations(FunctionDefTarget)

	moduleAST = ast.Module(body=cast(List[ast.stmt], allImports.makeListAst() + [FunctionDefTarget]), type_ignores=[])
	ast.fix_missing_locations(moduleAST)
	return ast.unparse(moduleAST)

def makeStrRLEcompacted(arrayTarget: numpy.ndarray, identifierName: Optional[str]=None) -> str:
	"""Converts a NumPy array into a compressed string representation using run-length encoding (RLE).

	This function takes a NumPy array and converts it into an optimized string representation by:
	1. Compressing consecutive sequences of numbers into range objects
	2. Minimizing repeated zeros using array multiplication syntax
	3. Converting the result into a valid Python array initialization statement

	Parameters:
		arrayTarget (numpy.ndarray): The input NumPy array to be converted
		identifierName (str): The variable name to use in the output string

	Returns:
		str: A string containing Python code that recreates the input array in compressed form.
			Format: "{identifierName} = numpy.array({compressed_data}, dtype=numpy.{dtype})"
	"""

	def compressRangesNDArrayNoFlatten(arraySlice):
		if isinstance(arraySlice, numpy.ndarray) and arraySlice.ndim > 1:
			return [compressRangesNDArrayNoFlatten(arraySlice[index]) for index in range(arraySlice.shape[0])]
		elif isinstance(arraySlice, numpy.ndarray) and arraySlice.ndim == 1:
			listWithRanges = []
			for group in more_itertools.consecutive_groups(arraySlice.tolist()):
				ImaSerious = list(group)
				if len(ImaSerious) <= 4:
					listWithRanges += ImaSerious
				else:
					ImaRange = [range(ImaSerious[0], ImaSerious[-1] + 1)]
					listWithRanges += ImaRange
			return listWithRanges
		return arraySlice

	arrayAsNestedLists = compressRangesNDArrayNoFlatten(arrayTarget)

	stringMinimized = python_minifier.minify(str(arrayAsNestedLists))
	commaZeroMaximum = arrayTarget.shape[-1] - 1
	stringMinimized = stringMinimized.replace('[0' + ',0'*commaZeroMaximum + ']', '[0]*'+str(commaZeroMaximum+1))
	for countZeros in range(commaZeroMaximum, 2, -1):
		stringMinimized = stringMinimized.replace(',0'*countZeros + ']', ']+[0]*'+str(countZeros))

	stringMinimized = stringMinimized.replace('range', '*range')

	if identifierName:
		return f"{identifierName} = array({stringMinimized}, dtype={arrayTarget.dtype})"
	return stringMinimized

def moveArrayTo_body(FunctionDefTarget: ast.FunctionDef, astArg: ast.arg, arrayTarget: numpy.ndarray, allImports: UniversalImportTracker) -> Tuple[ast.FunctionDef, UniversalImportTracker]:
	arrayType = type(arrayTarget)
	moduleConstructor = arrayType.__module__
	constructorName = arrayType.__name__
	# NOTE hack
	constructorName = constructorName.replace('ndarray', 'array')
	argData_dtype: numpy.dtype = arrayTarget.dtype
	argData_dtypeName = arrayTarget.dtype.name

	allImports.addImportFromStr(moduleConstructor, constructorName)
	allImports.addImportFromStr(moduleConstructor, argData_dtypeName)

	onlyDataRLE = makeStrRLEcompacted(arrayTarget)
	astStatement = cast(ast.Expr, ast.parse(onlyDataRLE).body[0])
	dataAst = astStatement.value

	arrayCall = Then.make_astCall(name=constructorName, args=[dataAst], dictionaryKeywords={'dtype': ast.Name(id=argData_dtypeName, ctx=ast.Load())})

	assignment = ast.Assign(targets=[ast.Name(id=astArg.arg, ctx=ast.Store())], value=arrayCall)
	FunctionDefTarget.body.insert(0, assignment)
	return FunctionDefTarget, allImports

def evaluateArrayIn_body(FunctionDefTarget: ast.FunctionDef, astArg: ast.arg, arrayTarget: numpy.ndarray, allImports: UniversalImportTracker) -> Tuple[ast.FunctionDef, UniversalImportTracker]:
	arrayType = type(arrayTarget)
	moduleConstructor = arrayType.__module__
	constructorName = arrayType.__name__
	# NOTE hack
	constructorName = constructorName.replace('ndarray', 'array')
	allImports.addImportFromStr(moduleConstructor, constructorName)

	for stmt in FunctionDefTarget.body.copy():
		if isinstance(stmt, ast.Assign):
			if isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Subscript):
				astAssignee: ast.Name = stmt.targets[0]
				argData_dtypeName = hackSSOTdatatype(astAssignee.id)
				allImports.addImportFromStr(moduleConstructor, argData_dtypeName)
				astSubscript: ast.Subscript = stmt.value
				if isinstance(astSubscript.value, ast.Name) and astSubscript.value.id == astArg.arg and isinstance(astSubscript.slice, ast.Attribute):
					indexAs_astAttribute: ast.Attribute = astSubscript.slice
					indexAsStr = ast.unparse(indexAs_astAttribute)
					argDataSlice = arrayTarget[eval(indexAsStr)]

					onlyDataRLE = makeStrRLEcompacted(argDataSlice)
					astStatement = cast(ast.Expr, ast.parse(onlyDataRLE).body[0])
					dataAst = astStatement.value

					arrayCall = Then.make_astCall(name=constructorName, args=[dataAst], dictionaryKeywords={'dtype': ast.Name(id=argData_dtypeName, ctx=ast.Load())})

					assignment = ast.Assign( targets=[astAssignee], value=arrayCall )
					FunctionDefTarget.body.insert(0, assignment)
					FunctionDefTarget.body.remove(stmt)
	return FunctionDefTarget, allImports

def evaluate_argIn_body(FunctionDefTarget: ast.FunctionDef, astArg: ast.arg, arrayTarget: numpy.ndarray, Z0Z_listChaff: List[str], allImports: UniversalImportTracker) -> Tuple[ast.FunctionDef, UniversalImportTracker]:
	moduleConstructor = Z0Z_getDatatypeModuleScalar()
	for stmt in FunctionDefTarget.body.copy():
		if isinstance(stmt, ast.Assign):
			if isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Subscript):
				astAssignee: ast.Name = stmt.targets[0]
				argData_dtypeName = hackSSOTdatatype(astAssignee.id)
				allImports.addImportFromStr(moduleConstructor, argData_dtypeName)
				astSubscript: ast.Subscript = stmt.value
				if isinstance(astSubscript.value, ast.Name) and astSubscript.value.id == astArg.arg and isinstance(astSubscript.slice, ast.Attribute):
					indexAs_astAttribute: ast.Attribute = astSubscript.slice
					indexAsStr = ast.unparse(indexAs_astAttribute)
					argDataSlice: int = arrayTarget[eval(indexAsStr)].item()
					astCall = ast.Call(func=ast.Name(id=argData_dtypeName, ctx=ast.Load()) , args=[ast.Constant(value=argDataSlice)], keywords=[])
					assignment = ast.Assign(targets=[astAssignee], value=astCall)
					if astAssignee.id not in Z0Z_listChaff:
						FunctionDefTarget.body.insert(0, assignment)
					FunctionDefTarget.body.remove(stmt)
	return FunctionDefTarget, allImports

def removeIdentifierAssignFrom_body(FunctionDefTarget: ast.FunctionDef, identifier) -> ast.FunctionDef:
	for stmt in FunctionDefTarget.body.copy():
		if isinstance(stmt, ast.Assign):
			if isinstance(stmt.targets[0], ast.Subscript) and isinstance(stmt.targets[0].value, ast.Name):
				if stmt.targets[0].value.id == identifier:
					FunctionDefTarget.body.remove(stmt)
	return FunctionDefTarget

def evaluateAnnAssignIn_body(FunctionDefTarget: ast.FunctionDef, allImports: UniversalImportTracker) -> Tuple[ast.FunctionDef, UniversalImportTracker]:
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

def astObjectToAstConstant(FunctionDefTarget: ast.FunctionDef, object: str, value: int) -> ast.FunctionDef:
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

def astNameToAstConstant(FunctionDefTarget: ast.FunctionDef, name: str, value: int) -> ast.FunctionDef:
	def findName(node: ast.AST) -> bool:
		return isinstance(node, ast.Name) and node.id == name

	def replaceWithConstant(node: ast.AST) -> ast.AST:
		return ast.copy_location(ast.Constant(value=value), node)

	return cast(ast.FunctionDef, NodeReplacer(findName, replaceWithConstant).visit(FunctionDefTarget))

def makeLauncherJobNumba(callableTarget: str, pathFilenameFoldsTotal: pathlib.Path) -> ast.Module:
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

def addReturnJobNumba(FunctionDefTarget: ast.FunctionDef, stateJob: computationState, allImports: UniversalImportTracker) -> Tuple[ast.FunctionDef, UniversalImportTracker]:
	"""Add multiplication and return statement to function, properly constructing AST nodes."""
	# Create AST for multiplication operation
	multiplicand = Z0Z_identifierCountFolds
	datatype = hackSSOTdatatype(multiplicand)
	multiplyOperation = ast.BinOp(
		left=ast.Name(id=multiplicand, ctx=ast.Load()),
		op=ast.Mult(), right=ast.Constant(value=int(stateJob['foldGroups'][-1])))

	returnStatement = ast.Return(value=multiplyOperation)

	datatypeModuleScalar = Z0Z_getDatatypeModuleScalar()
	allImports.addImportFromStr(datatypeModuleScalar, datatype)
	FunctionDefTarget.returns = ast.Name(id=datatype, ctx=ast.Load())

	FunctionDefTarget.body.append(returnStatement)

	return FunctionDefTarget, allImports

def unrollWhileLoop(FunctionDefTarget: ast.FunctionDef, iteratorName: str, iterationsTotal: int, connectionGraph: numpy.ndarray[Tuple[int, int, int], numpy.dtype[numpy.integer[Any]]]) -> ast.FunctionDef:
	"""
	Unroll the countGaps loop: in theDao, it is a while loop, of course.
	However, it could be written as `for indexDimension in range(dimensionsTotal):`.
	It is useful to note that it could also be written as `for indexDimension in range(connectionGraph.shape[0]):`.
	We will unroll the loop into a series of stateJob['my'][indexMy.dimensionsTotal]-many code blocks that are similar but not identical.
	In each code block, we know the value of the identifier `indexDimension`, so we replace the identifier with its value.
	Furthermore, we will split connectionGraph into arrays along the first axis.
	`connectionGraph[indexDimension, leaf1ndex, leafBelow[leafConnectee]]`
	`connectionGraph0[leaf1ndex, leafBelow[leafConnectee]]`
	`connectionGraph1[leaf1ndex, leafBelow[leafConnectee]]`
	`connectionGraphN[leaf1ndex, leafBelow[leafConnectee]]`

	After unrolling, we can remove three `indexDimension` statements: 1) the first initialization, which is really a memory allocation, 2) the loop initialization, and 3) the loop increment.
	"""
	return FunctionDefTarget

def Z0Z_UnhandledDecorators(astCallable: ast.FunctionDef) -> ast.FunctionDef:
	# TODO: more explicit handling of decorators. I'm able to ignore this because I know `algorithmSource` doesn't have any decorators.
	for decoratorItem in astCallable.decorator_list.copy():
		import warnings
		astCallable.decorator_list.remove(decoratorItem)
		warnings.warn(f"Removed decorator {ast.unparse(decoratorItem)} from {astCallable.name}")
	return astCallable

def thisIsNumbaDotJit(Ima: ast.AST) -> bool:
	return ifThis.isCallWithAttribute(Z0Z_getDatatypeModuleScalar(), Z0Z_getDecoratorCallable())(Ima)

def thisIsJit(Ima: ast.AST) -> bool:
	return ifThis.isCallWithName(Z0Z_getDecoratorCallable())(Ima)

def thisIsAnyNumbaJitDecorator(Ima: ast.AST) -> bool:
	return thisIsNumbaDotJit(Ima) or thisIsJit(Ima)

def decorateCallableWithNumba(FunctionDefTarget: ast.FunctionDef, allImports: UniversalImportTracker, parametersNumba: Optional[ParametersNumba]=None) -> Tuple[ast.FunctionDef, UniversalImportTracker]:
	datatypeModuleDecorator = Z0Z_getDatatypeModuleScalar()
	def make_arg4parameter(signatureElement: ast.arg):
		if isinstance(signatureElement.annotation, ast.Subscript) and isinstance(signatureElement.annotation.slice, ast.Tuple):
			annotationShape = signatureElement.annotation.slice.elts[0]
			if isinstance(annotationShape, ast.Subscript) and isinstance(annotationShape.slice, ast.Tuple):
				shapeAsListSlices: Sequence[ast.expr] = [ast.Slice() for axis in range(len(annotationShape.slice.elts))]
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

	list_argsDecorator: Sequence[ast.expr] = []

	list_arg4signature_or_function: Sequence[ast.expr] = []
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
