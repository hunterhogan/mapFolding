from collections.abc import Callable, Sequence
from mapFolding import EnumIndices
from typing import Any, cast, NamedTuple
from typing import TypeAlias
from Z0Z_tools import updateExtendPolishDictionaryLists
import ast
import collections
import pathlib

ast_Identifier: TypeAlias = str

class YouOughtaKnow(NamedTuple):
	callableSynthesized: str
	pathFilenameForMe: pathlib.Path
	astForCompetentProgrammers: ast.ImportFrom

class ifThis:
	@staticmethod
	def nameIs(allegedly: str) -> Callable[[ast.AST], bool]:
		return lambda node: (isinstance(node, ast.Name) and node.id == allegedly)

	@staticmethod
	def subscriptNameIs(allegedly: str) -> Callable[[ast.AST], bool]:
		return lambda node: (isinstance(node, ast.Subscript)
							and isinstance(node.value, ast.Name)
							and node.value.id == allegedly)

	@staticmethod
	def NameReallyIs(allegedly: str) -> Callable[[ast.AST], bool]:
		return ifThis.anyOf(ifThis.nameIs(allegedly), ifThis.subscriptNameIs(allegedly))

	@staticmethod
	def CallAsNameIs(callableName: str) -> Callable[[ast.AST], bool]:
		return lambda node: (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == callableName)

	@staticmethod
	def CallAsModuleAttributeIs(moduleName: str, callableName: str) -> Callable[[ast.AST], bool]:
		return lambda node: (isinstance(node, ast.Call)
							and isinstance(node.func, ast.Attribute)
							and isinstance(node.func.value, ast.Name)
							and node.func.value.id == moduleName
							and node.func.attr == callableName)

	@staticmethod
	def CallReallyIs(moduleName: str, callableName: str) -> Callable[[ast.AST], bool]:
		return ifThis.anyOf(ifThis.CallAsNameIs(callableName), ifThis.CallAsModuleAttributeIs(moduleName, callableName))

	@staticmethod
	def CallDoesNotCallItself(moduleName: str, callableName: str) -> Callable[[ast.AST], bool]:
		return lambda node: (ifThis.CallReallyIs(moduleName, callableName)(node)
							and 1 == sum(1 for descendant in ast.walk(node)
											if ifThis.CallReallyIs(moduleName, callableName)(descendant)))

	@staticmethod
	def RecklessCallAsAttributeIs(callableName: str) -> Callable[[ast.AST], bool]:
		"""Warning: You might match more than you want."""
		return lambda node: (isinstance(node, ast.Call)
							and isinstance(node.func, ast.Attribute)
							and isinstance(node.func.value, ast.Name)
							and node.func.attr == callableName)

	@staticmethod
	def RecklessCallReallyIs(callableName: str) -> Callable[[ast.AST], bool]:
		"""Warning: You might match more than you want."""
		return ifThis.anyOf(ifThis.CallAsNameIs(callableName), ifThis.RecklessCallAsAttributeIs(callableName))

	@staticmethod
	def AssignTo(identifier: str) -> Callable[[ast.AST], bool]:
		return lambda node: (isinstance(node, ast.Assign)
								and len(node.targets) > 0
								and ifThis.NameReallyIs(identifier)(node.targets[0]))

	@staticmethod
	def AnnAssignTo(identifier: str) -> Callable[[ast.AST], bool]:
		return lambda node: (isinstance(node, ast.AnnAssign)
								and ifThis.NameReallyIs(identifier)(node.target))

	@staticmethod
	def AugAssignTo(identifier: str) -> Callable[[ast.AST], bool]:
		return lambda node: (isinstance(node, ast.AugAssign)
								and ifThis.NameReallyIs(identifier)(node.target))

	@staticmethod
	def anyAssignmentTo(identifier: str) -> Callable[[ast.AST], bool]:
		return ifThis.anyOf(ifThis.AssignTo(identifier), ifThis.AnnAssignTo(identifier), ifThis.AugAssignTo(identifier))

	@staticmethod
	def anyOf(*predicates: Callable[[ast.AST], bool]) -> Callable[[ast.AST], bool]:
		return lambda node: any(pred(node) for pred in predicates)

	@staticmethod
	def isUnpackingAnArray(identifier:str) -> Callable[[ast.AST], bool]:
		return lambda node: (isinstance(node, ast.Assign)
						and  isinstance(node.targets[0], ast.Name)
						and  isinstance(node.value, ast.Subscript)
						and  isinstance(node.value.value, ast.Name)
						and  node.value.value.id == identifier
						and  isinstance(node.value.slice, ast.Attribute)
						)

class Then:
	@staticmethod
	def copy_astCallKeywords(astCall: ast.Call) -> dict[str, Any]:
		"""Extract keyword parameters from a decorator AST node."""
		dictionaryKeywords: dict[str, Any] = {}
		for keywordItem in astCall.keywords:
			if isinstance(keywordItem.value, ast.Constant) and keywordItem.arg is not None:
				dictionaryKeywords[keywordItem.arg] = keywordItem.value.value
		return dictionaryKeywords

	@staticmethod
	def make_astCall(caller: ast.Name | ast.Attribute
				, args: Sequence[ast.expr] | None = None
				, list_astKeywords: Sequence[ast.keyword] | None = None
				, dictionaryKeywords: dict[str, Any] | None = None
				) -> ast.Call:
		list_dictionaryKeywords: list[ast.keyword] = [ast.keyword(arg=keyName, value=ast.Constant(value=keyValue)) for keyName, keyValue in dictionaryKeywords.items()] if dictionaryKeywords else []
		return ast.Call(
			func=caller,
			args=list(args) if args else [],
			keywords=list_dictionaryKeywords + list(list_astKeywords) if list_astKeywords else [],
		)

	@staticmethod
	def makeName(identifier: ast_Identifier) -> ast.Name:
		return ast.Name(id=identifier, ctx=ast.Load())

	@staticmethod
	def addDOTname(nameChain: ast.Name | ast.Attribute, dotName: str) -> ast.Attribute:
		return ast.Attribute(value=nameChain, attr=dotName, ctx=ast.Load())

	@staticmethod
	def makeNameDOTname(identifier: ast_Identifier, *dotName: str) -> ast.Name | ast.Attribute:
		nameDOTname: ast.Name | ast.Attribute = Then.makeName(identifier)
		if not dotName:
			return nameDOTname
		for suffix in dotName:
			nameDOTname = Then.addDOTname(nameDOTname, suffix)
		return nameDOTname

	@staticmethod
	def appendThis(astStatement: ast.AST) -> Callable[[ast.AST], Sequence[ast.stmt]]:
		return lambda belowMe: [cast(ast.stmt, belowMe),
								cast(ast.stmt, astStatement)]

	@staticmethod
	def replaceWith(astStatement: ast.AST) -> Callable[[ast.AST], ast.stmt]:
		return lambda replaceMe: cast(ast.stmt, astStatement)

	@staticmethod
	def removeThis(astNode: ast.AST) -> None:
		return None

class NodeReplacer(ast.NodeTransformer):
	"""
	A node transformer that replaces or removes AST nodes based on a condition.
	This transformer traverses an AST and for each node checks a predicate. If the predicate
	returns True, the transformer uses the replacement builder to obtain a new node. Returning
	None from the replacement builder indicates that the node should be removed.

	Attributes:
		findMe: A function that finds all locations that match a one or more conditions.
		doThis: A function that does work at each location, such as make a new node, collect information or delete the node.

	Methods:
		visit(node: ast.AST) -> Optional[ast.AST]:
			Visits each node in the AST, replacing or removing it based on the predicate.
	"""
	def __init__(self, findMe: Callable[[ast.AST], bool], doThis: Callable[[ast.AST], ast.AST | Sequence[ast.AST] | None]) -> None:
		self.findMe: Callable[[ast.AST], bool] = findMe
		self.doThis: Callable[[ast.AST], ast.AST | Sequence[ast.AST] | None] = doThis

	def visit(self, node: ast.AST) -> ast.AST | Sequence[ast.AST] | None:
		if self.findMe(node):
			return self.doThis(node)
		return super().visit(node)

class UniversalImportTracker:
	def __init__(self, startWith: ast.AST | None = None) -> None:
		self.dictionaryImportFrom: dict[str, set[tuple[str, str | None]]] = collections.defaultdict(set)
		self.setImport: set[str] = set()

		if startWith:
			self.walkThis(startWith)

	def addAst(self, astImport_: ast.Import | ast.ImportFrom) -> None:
		if isinstance(astImport_, ast.Import):
			for alias in astImport_.names:
				self.setImport.add(alias.name)
		elif isinstance(astImport_, ast.ImportFrom): # type: ignore
			if astImport_.module is not None:
				self.dictionaryImportFrom[astImport_.module].update((alias.name, alias.asname) for alias in astImport_.names)

	def addImportStr(self, module: str) -> None:
		self.setImport.add(module)

	def addImportFromStr(self, module: str, name: str, asname: str | None = None) -> None:
		self.dictionaryImportFrom[module].add((name, asname))

	def makeListAst(self) -> list[ast.ImportFrom | ast.Import]:
		listAstImportFrom: list[ast.ImportFrom] = []
		for module, setOfNameTuples in sorted(self.dictionaryImportFrom.items()):
			listAliases: list[ast.alias] = []
			for name, asname in setOfNameTuples:
				listAliases.append(ast.alias(name=name, asname=asname))
			listAstImportFrom.append(ast.ImportFrom(module=module, names=listAliases, level=0))

		listAstImport: list[ast.Import] = [ast.Import(names=[ast.alias(name=name, asname=None)]) for name in sorted(self.setImport)]
		return listAstImportFrom + listAstImport

	def update(self, *fromTracker: 'UniversalImportTracker') -> None:
		"""
		Update this tracker with imports from one or more other trackers.

		Parameters:
			*fromTracker: One or more UniversalImportTracker objects to merge from.
		"""
		# Merge all import-from dictionaries
		dictionaryMerged: dict[str, list[Any]] = updateExtendPolishDictionaryLists(self.dictionaryImportFrom, *(tracker.dictionaryImportFrom for tracker in fromTracker), destroyDuplicates=True, reorderLists=True)

		# Convert lists back to sets for each module's imports
		self.dictionaryImportFrom = {module: set(listNames) for module, listNames in dictionaryMerged.items()}

		# Update direct imports
		for tracker in fromTracker:
			self.setImport.update(tracker.setImport)

	def walkThis(self, walkThis: ast.AST) -> None:
			for smurf in ast.walk(walkThis):
				if isinstance(smurf, (ast.Import, ast.ImportFrom)):
					self.addAst(smurf)

class FunctionInliner(ast.NodeTransformer):
	def __init__(self, dictionaryFunctions: dict[str, ast.FunctionDef]) -> None:
		self.dictionaryFunctions: dict[str, ast.FunctionDef] = dictionaryFunctions

	def inlineFunctionBody(self, callableTargetName: str) -> ast.FunctionDef:
		inlineDefinition: ast.FunctionDef = self.dictionaryFunctions[callableTargetName]
		# Process nested calls within the inlined function
		for astNode in ast.walk(inlineDefinition):
			self.visit(astNode)
		return inlineDefinition

	def visit_Call(self, node: ast.Call) -> Any | ast.Constant | ast.Call | ast.AST:
		callNodeVisited: ast.AST = self.generic_visit(node)
		if (isinstance(callNodeVisited, ast.Call)
		and isinstance(callNodeVisited.func, ast.Name)
		and callNodeVisited.func.id in self.dictionaryFunctions
		and ifThis.CallDoesNotCallItself("", callNodeVisited.func.id)(callNodeVisited)):
			inlineDefinition: ast.FunctionDef = self.inlineFunctionBody(callNodeVisited.func.id)

			if (inlineDefinition and inlineDefinition.body):
				statementTerminating: ast.stmt = inlineDefinition.body[-1]

				if (isinstance(statementTerminating, ast.Return)
				and statementTerminating.value is not None):
					return self.visit(statementTerminating.value)
				elif isinstance(statementTerminating, ast.Expr):
					return self.visit(statementTerminating.value)
				else:
					return ast.Constant(value=None)
		return callNodeVisited

	def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.AST]:
		if isinstance(node.value, ast.Call):
			if (isinstance(node.value.func, ast.Name)
			and node.value.func.id in self.dictionaryFunctions
			and ifThis.CallDoesNotCallItself("", node.value.func.id)(node.value)):
				inlineDefinition: ast.FunctionDef = self.inlineFunctionBody(node.value.func.id)
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

	def __init__(self, enumIndexClass: type[EnumIndices], arrayName: str) -> None:
		self.enumIndexClass = enumIndexClass
		self.arrayName = arrayName
		self.substitutions: dict[str, Any] = {}

	def extract_member_name(self, node: ast.AST) -> str | None:
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
				return ast.Tuple(elts=cast(list[ast.expr], [self.transform_slice_element(elt) for elt in node.elts]), ctx=node.ctx)
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
			firstElement: ast.expr = node.slice.elts[0]
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

		initializations: list[ast.Assign] = []
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
