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
import copy
import inspect
import more_itertools
import numba
import numpy
import os
import pathlib
import python_minifier

youOughtaKnow = collections.namedtuple('youOughtaKnow', ['callableSynthesized', 'pathFilenameForMe', 'astForCompetentProgrammers'])

class ifThis:
	"""Generic AST node predicate builder."""
	@staticmethod
	def nameIs(allegedly: str) -> Callable[[ast.AST], bool]:
		return lambda node: (isinstance(node, ast.Name) and node.id == allegedly)

	@staticmethod
	def isCallWithAttribute(moduleName: str, callableName: str) -> Callable[[ast.AST], bool]:
		return lambda node: (isinstance(node, ast.Call)
							and isinstance(node.func, ast.Attribute)
							and isinstance(node.func.value, ast.Name)
							and node.func.value.id == moduleName
							and node.func.attr == callableName)

	@staticmethod
	def isCallWithName(callableName: str) -> Callable[[ast.AST], bool]:
		return lambda node: (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == callableName)

	@staticmethod
	def anyOf(*predicates: Callable[[ast.AST], bool]) -> Callable[[ast.AST], bool]:
		return lambda node: any(pred(node) for pred in predicates)

class Then:
	"""Generic actions."""
	@staticmethod
	def copy_astCallKeywords(astCall: ast.Call) -> Dict[str, Any]:
		"""Extract keyword parameters from a decorator AST node."""
		dictionaryKeywords: Dict[str, Any] = {}
		for keywordItem in astCall.keywords:
			if isinstance(keywordItem.value, ast.Constant) and keywordItem.arg is not None:
				dictionaryKeywords[keywordItem.arg] = keywordItem.value.value
		return dictionaryKeywords

	@staticmethod
	def make_astCall(name: str, args: Optional[Sequence[ast.expr]]=None, list_astKeywords: Optional[Sequence[ast.keyword]]=None, dictionaryKeywords: Optional[Dict[str, Any]]=None) -> ast.Call:
		list_dictionaryKeywords = [ast.keyword(arg=keyName, value=ast.Constant(value=keyValue)) for keyName, keyValue in dictionaryKeywords.items()] if dictionaryKeywords else []
		return ast.Call(
			func=ast.Name(id=name, ctx=ast.Load()),
			args=list(args) if args else [],
			keywords=list_dictionaryKeywords + list(list_astKeywords) if list_astKeywords else [],
		)

class NodeReplacer(ast.NodeTransformer):
	"""
	A node transformer that replaces or removes AST nodes based on a condition.
	This transformer traverses an AST and for each node checks a predicate. If the predicate
	returns True, the transformer uses the replacement builder to obtain a new node. Returning
	None from the replacement builder indicates that the node should be removed.

	Attributes:
		findMe (Callable[[ast.AST], bool]): A function that determines whether a node should be replaced.
		nodeReplacementBuilder (Callable[[ast.AST], Optional[ast.AST]]): A function that returns a new node
			or None to remove the node.

	Methods:
		visit(node: ast.AST) -> Optional[ast.AST]:
			Visits each node in the AST, replacing or removing it based on the predicate.
	"""
	def __init__(self, findMe: Callable[[ast.AST], bool], nodeReplacementBuilder: Callable[[ast.AST], Optional[ast.AST]]) -> None:
		self.findMe = findMe
		self.nodeReplacementBuilder = nodeReplacementBuilder

	def visit(self, node: ast.AST) -> Optional[ast.AST]:
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

class RecursiveInliner(ast.NodeTransformer):
	"""
	Class RecursiveInliner:
		A custom AST NodeTransformer designed to recursively inline function calls from a given dictionary
		of function definitions into the AST. Once a particular function has been inlined, it is marked
		as completed to avoid repeated inlining. This transformation modifies the AST in-place by substituting
		eligible function calls with the body of their corresponding function.
		Attributes:
			dictionaryFunctions (Dict[str, ast.FunctionDef]):
				A mapping of function name to its AST definition, used as a source for inlining.
			callablesCompleted (Set[str]):
				A set to track function names that have already been inlined to prevent multiple expansions.
		Methods:
			inlineFunctionBody(callableTargetName: str) -> Optional[ast.FunctionDef]:
				Retrieves the AST definition for a given function name from dictionaryFunctions
				and recursively inlines any function calls within it. Returns the function definition
				that was inlined or None if the function was already processed.
			visit_Call(callNode: ast.Call) -> ast.AST:
				Inspects calls within the AST. If a function call matches one in dictionaryFunctions,
				it is replaced by the inlined body. If the last statement in the inlined body is a return
				or an expression, that value or expression is substituted; otherwise, a constant is returned.
			visit_Expr(node: ast.Expr) -> Union[ast.AST, List[ast.AST]]:
				Handles expression nodes in the AST. If the expression is a function call from
				dictionaryFunctions, its statements are expanded in place, effectively inlining
				the called function's statements into the surrounding context.
	"""
	def __init__(self, dictionaryFunctions: Dict[str, ast.FunctionDef]):
		self.dictionaryFunctions = dictionaryFunctions
		self.callablesCompleted: Set[str] = set()

	def inlineFunctionBody(self, callableTargetName: str) -> Optional[ast.FunctionDef]:
		if (callableTargetName in self.callablesCompleted):
			return None

		self.callablesCompleted.add(callableTargetName)
		inlineDefinition = self.dictionaryFunctions[callableTargetName]
		for astNode in ast.walk(inlineDefinition):
			self.visit(astNode)
		return inlineDefinition

	def visit_Call(self, node: ast.Call) -> ast.AST:
		callNodeVisited = self.generic_visit(node)
		if (isinstance(callNodeVisited, ast.Call) and isinstance(callNodeVisited.func, ast.Name) and callNodeVisited.func.id in self.dictionaryFunctions):
			inlineDefinition = self.inlineFunctionBody(callNodeVisited.func.id)
			if (inlineDefinition and inlineDefinition.body):
				statementTerminating = inlineDefinition.body[-1]
				if (isinstance(statementTerminating, ast.Return) and statementTerminating.value is not None):
					return self.visit(statementTerminating.value)
				elif (isinstance(statementTerminating, ast.Expr) and statementTerminating.value is not None):
					return self.visit(statementTerminating.value)
				return ast.Constant(value=None)
		return callNodeVisited

	def visit_Expr(self, node: ast.Expr) -> Union[ast.AST, List[ast.AST]]:
		if (isinstance(node.value, ast.Call)):
			if (isinstance(node.value.func, ast.Name) and node.value.func.id in self.dictionaryFunctions):
				inlineDefinition = self.inlineFunctionBody(node.value.func.id)
				if (inlineDefinition):
					return [self.visit(stmt) for stmt in inlineDefinition.body]
		return self.generic_visit(node)

class UnpackArrays(ast.NodeTransformer):
	"""
	A class that transforms array accesses using enum indices into local variables.

	This AST transformer identifies array accesses using enum indices and replaces them
	with local variables, adding initialization statements at the start of functions.

	Parameters:
		enumIndexClass (Type[EnumIndices]): The enum class used for array indexing
		arrayName (str): The name of the array being accessed

	Attributes:
		enumIndexClass (Type[EnumIndices]): Stored enum class for index lookups
		arrayName (str): Name of the array being transformed
		substitutions (dict): Tracks variable substitutions and their original nodes

	The transformer handles two main cases:
	1. Scalar array access - array[EnumIndices.MEMBER]
	2. Array slice access - array[EnumIndices.MEMBER, other_indices...]
	For each identified access pattern, it:
	1. Creates a local variable named after the enum member
	2. Adds initialization code at function start
	3. Replaces original array access with the local variable
	"""

	def __init__(self, enumIndexClass: Type[EnumIndices], arrayName: str):
		self.enumIndexClass = enumIndexClass
		self.arrayName = arrayName
		self.substitutions = {}

	def extract_member_name(self, node: ast.AST) -> Optional[str]:
		"""Recursively extract enum member name from any node in the AST."""
		if isinstance(node, ast.Attribute) and node.attr == 'value':
			innerAttribute = node.value
			while isinstance(innerAttribute, ast.Attribute):
				if (isinstance(innerAttribute.value, ast.Name) and innerAttribute.value.id == self.enumIndexClass.__name__):
					return innerAttribute.attr
				innerAttribute = innerAttribute.value
		return None

	def transform_slice_element(self, node: ast.AST) -> ast.AST:
		"""Transform any enum references within a slice element."""
		if isinstance(node, ast.Subscript):
			if isinstance(node.slice, ast.Attribute):
				member_name = self.extract_member_name(node.slice)
				if member_name:
					return ast.Name(id=member_name, ctx=node.ctx)
			elif isinstance(node, ast.Tuple):
				# Handle tuple slices by transforming each element
				return ast.Tuple(elts=cast(List[ast.expr], [self.transform_slice_element(elt) for elt in node.elts]), ctx=node.ctx)
		elif isinstance(node, ast.Attribute):
			member_name = self.extract_member_name(node)
			if member_name:
				return ast.Name(id=member_name, ctx=ast.Load())
		return node

	def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
		# Recursively visit any nested subscripts in value or slice
		node.value = self.visit(node.value)
		node.slice = self.visit(node.slice)
		# If node.value is not our arrayName, just return node
		if not (isinstance(node.value, ast.Name) and node.value.id == self.arrayName):
			return node

		# Handle scalar array access
		if isinstance(node.slice, ast.Attribute):
			memberName = self.extract_member_name(node.slice)
			if memberName:
				self.substitutions[memberName] = ('scalar', node)
				return ast.Name(id=memberName, ctx=ast.Load())

		# Handle array slice access
		if isinstance(node.slice, ast.Tuple) and node.slice.elts:
			firstElement = node.slice.elts[0]
			memberName = self.extract_member_name(firstElement)
			sliceRemainder = [self.visit(elem) for elem in node.slice.elts[1:]]
			if memberName:
				self.substitutions[memberName] = ('array', node)
				if len(sliceRemainder) == 0:
					return ast.Name(id=memberName, ctx=ast.Load())
				return ast.Subscript(value=ast.Name(id=memberName, ctx=ast.Load()), slice=ast.Tuple(elts=sliceRemainder, ctx=ast.Load()) if len(sliceRemainder) > 1 else sliceRemainder[0], ctx=ast.Load())

		# If single-element tuple, unwrap
		if isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 1:
			node.slice = node.slice.elts[0]

		return node

	def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
		node = cast(ast.FunctionDef, self.generic_visit(node))

		initializations = []
		for name, (kind, original_node) in self.substitutions.items():
			if kind == 'scalar':
				initializations.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=original_node))
			else:  # array
				initializations.append(
					ast.Assign(
						targets=[ast.Name(id=name, ctx=ast.Store())],
						value=ast.Subscript(value=ast.Name(id=self.arrayName, ctx=ast.Load()),
							slice=ast.Attribute(value=ast.Attribute(
									value=ast.Name(id=self.enumIndexClass.__name__, ctx=ast.Load()),
									attr=name, ctx=ast.Load()), attr='value', ctx=ast.Load()), ctx=ast.Load())))

		node.body = initializations + node.body
		return node
