from mapFolding import (
	computationState,
	EnumIndices,
	FREAKOUT,
	getAlgorithmSource,
	getFilenameFoldsTotal,
	getPathFilenameFoldsTotal,
	getPathJobRootDEFAULT,
	getPathSyntheticModules,
	hackSSOTdatatype,
	indexMy,
	indexTrack,
	moduleOfSyntheticModules,
	myPackageNameIs,
	ParametersNumba,
	parametersNumbaDEFAULT,
	parametersNumbaFailEarly,
	parametersNumbaSuperJit,
	parametersNumbaSuperJitParallel,
	setDatatypeElephino,
	setDatatypeFoldsTotal,
	setDatatypeLeavesTotal,
	setDatatypeModule,
	Z0Z_getDatatypeModuleScalar,
	Z0Z_getDecoratorCallable,
	Z0Z_identifierCountFolds,
	Z0Z_setDatatypeModuleScalar,
	Z0Z_setDecoratorCallable,
)
from mapFolding.someAssemblyRequired import makeStateJob
from types import ModuleType
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Set, Tuple, Type, Union
import ast
import autoflake
import collections
import inspect
import more_itertools
import numba
import numpy
import os
import pathlib
import python_minifier

youOughtaKnow = collections.namedtuple('youOughtaKnow', ['callableSynthesized', 'pathFilenameForMe', 'astForCompetentProgrammers'])

class NodeReplacer(ast.NodeTransformer):
	"""Generic node replacement using configurable predicate and builder."""
	def __init__(self, findMe: Callable[[ast.AST], bool], nodeReplacementBuilder: Callable[[ast.AST], ast.AST]):
		self.findMe = findMe
		self.nodeReplacementBuilder = nodeReplacementBuilder

	def visit(self, node: ast.AST) -> ast.AST:
		if self.findMe(node):
			return self.nodeReplacementBuilder(node)
		return super().visit(node)

class UniversalImportTracker:
	def __init__(self):
		self.dictionaryImportFrom = collections.defaultdict(set)
		self.setImport = set()

	def addAst(self, astImport_: Union[ast.Import, ast.ImportFrom]) -> None:
		if isinstance(astImport_, ast.Import):
			for alias in astImport_.names:
				self.setImport.add(alias.name)
		elif isinstance(astImport_, ast.ImportFrom):
			self.dictionaryImportFrom[astImport_.module].update(alias.name for alias in astImport_.names)

	def addImportFromStr(self, module: str, name: str) -> None:
		self.dictionaryImportFrom[module].add(name)

	def addImportStr(self, name: str) -> None:
		self.setImport.add(name)

	def makeListAst(self) -> List[Union[ast.ImportFrom, ast.Import]]:
		listAstImportFrom = [ast.ImportFrom(module=module, names=[ast.alias(name=name, asname=None)], level=0) for module, names in self.dictionaryImportFrom.items() for name in names]
		listAstImport = [ast.Import(names=[ast.alias(name=name, asname=None)]) for name in self.setImport]
		return listAstImportFrom + listAstImport

def Z0Z_UnhandledDecorators(astCallable: ast.FunctionDef) -> ast.FunctionDef:
	# TODO: more explicit handling of decorators. I'm able to ignore this because I know `algorithmSource` doesn't have any decorators.
	for decoratorItem in astCallable.decorator_list.copy():
		import warnings
		astCallable.decorator_list.remove(decoratorItem)
		warnings.warn(f"Removed decorator {ast.unparse(decoratorItem)} from {astCallable.name}")
	return astCallable
def isThisNodeNumbaJitCall(whoAmI: ast.AST) -> bool:
	return (isinstance(whoAmI, ast.Call) and isinstance(whoAmI.func, ast.Attribute) and whoAmI.func.attr == Z0Z_getDecoratorCallable())
def isThisNodeJitCall(whoAmI: ast.AST) -> bool:
	return (isinstance(whoAmI, ast.Call) and isinstance(whoAmI.func, ast.Name) and whoAmI.func.id == Z0Z_getDecoratorCallable())
def isThisNodeNumbaJitDecorator(whoAmI: ast.AST) -> bool:
	return isThisNodeNumbaJitCall(whoAmI) or isThisNodeJitCall(whoAmI)
def Z0Z_generalizeThis(FunctionDefTarget: ast.FunctionDef, parametersNumba: Optional[ParametersNumba]=None) -> Tuple[ast.FunctionDef, ParametersNumba | None]:
	def recycleParametersNumba(decorator: ast.Call) -> Dict[str, Any]:
		parametersNumbaExtracted: Dict[str, Any] = {}
		for keywordItem in decorator.keywords:
			if isinstance(keywordItem.value, ast.Constant) and keywordItem.arg is not None:
				parametersNumbaExtracted[keywordItem.arg] = keywordItem.value.value
		return parametersNumbaExtracted

	for decorator in FunctionDefTarget.decorator_list.copy():
		if isThisNodeNumbaJitDecorator(decorator):
			decorator = cast(ast.Call, decorator)
			if parametersNumba is None:
				parametersNumbaSherpa = recycleParametersNumba(decorator)
				if (HunterIsSureThereAreBetterWaysToDoThis := True):
					if parametersNumbaSherpa:
						parametersNumba = cast(ParametersNumba, parametersNumbaSherpa)
		FunctionDefTarget.decorator_list.remove(decorator)

	return FunctionDefTarget, parametersNumba

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

	FunctionDefTarget, parametersNumba = Z0Z_generalizeThis(FunctionDefTarget, parametersNumba)
	FunctionDefTarget = Z0Z_UnhandledDecorators(FunctionDefTarget)
	if parametersNumba is None:
		parametersNumba = parametersNumbaDEFAULT
	listDecoratorKeywords = [ast.keyword(arg=parameterName, value=ast.Constant(value=parameterValue)) for parameterName, parameterValue in parametersNumba.items()]

	decoratorModule = Z0Z_getDatatypeModuleScalar()
	decoratorCallable = Z0Z_getDecoratorCallable()
	allImports.addImportFromStr(decoratorModule, decoratorCallable)
	astDecorator = ast.Call(
		func=ast.Name(id=decoratorCallable, ctx=ast.Load())
		, args=list_argsDecorator if list_argsDecorator else []
		, keywords=listDecoratorKeywords)

	FunctionDefTarget.decorator_list = [astDecorator]
	return FunctionDefTarget, allImports

def makeDecoratorJobNumba(FunctionDefTarget: ast.FunctionDef, allImports: UniversalImportTracker, parametersNumba: Optional[ParametersNumba]=None) -> Tuple[ast.FunctionDef, UniversalImportTracker]:
	decoratorCallable = Z0Z_getDecoratorCallable()
	def convertToPlainJit(node: ast.Call) -> ast.Call:
		node.func = ast.Name(id=decoratorCallable, ctx=ast.Load())
		return node

	FunctionDefTarget, parametersNumba = Z0Z_generalizeThis(FunctionDefTarget, parametersNumba)

	FunctionDefTarget, allImports = decorateCallableWithNumba(FunctionDefTarget, allImports, parametersNumba)
	if isThisNodeNumbaJitCall(FunctionDefTarget.decorator_list[0]):
		FunctionDefTarget.decorator_list[0] = convertToPlainJit(cast(ast.Call, FunctionDefTarget.decorator_list[0]))

	return FunctionDefTarget, allImports
