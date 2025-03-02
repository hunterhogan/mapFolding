from Z0Z_tools import updateExtendPolishDictionaryLists
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, cast, NamedTuple
from typing import TypeAlias
import ast

ast_Identifier: TypeAlias = str

# NOTE: this is weak
class YouOughtaKnow(NamedTuple):
	callableSynthesized: str
	pathFilenameForMe: Path
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
	def CallAsNameIsIn(container: Iterable[Any]) -> Callable[[ast.AST], bool]:
		return lambda node: (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in container)

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
	def isAnnAssign() -> Callable[[ast.AST], bool]:
		return lambda node: isinstance(node, ast.AnnAssign)

	@staticmethod
	def isAnnAssignTo(identifier: str) -> Callable[[ast.AST], bool]:
		return lambda node: (ifThis.isAnnAssign()(node)
								and ifThis.NameReallyIs(identifier)(node.target)) # type: ignore

	@staticmethod
	def AugAssignTo(identifier: str) -> Callable[[ast.AST], bool]:
		return lambda node: (isinstance(node, ast.AugAssign)
								and ifThis.NameReallyIs(identifier)(node.target))

	@staticmethod
	def anyAssignmentTo(identifier: str) -> Callable[[ast.AST], bool]:
		return ifThis.anyOf(ifThis.AssignTo(identifier), ifThis.isAnnAssignTo(identifier), ifThis.AugAssignTo(identifier))

	@staticmethod
	def anyOf(*predicates: Callable[[ast.AST], bool]) -> Callable[[ast.AST], bool]:
		return lambda node: any(pred(node) for pred in predicates)

	@staticmethod
	def is_dataclassesDOTField() -> Callable[[ast.AST], bool]: # type: ignore
		# dataclasses.Field
		pass

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
	def insertThisAbove(astStatement: ast.AST) -> Callable[[ast.AST], Sequence[ast.stmt]]:
		return lambda aboveMe: [cast(ast.stmt, astStatement),
								cast(ast.stmt, aboveMe)]

	@staticmethod
	def insertThisBelow(astStatement: ast.AST) -> Callable[[ast.AST], Sequence[ast.stmt]]:
		return lambda belowMe: [cast(ast.stmt, belowMe),
								cast(ast.stmt, astStatement)]

	@staticmethod
	def appendTo(primitiveList: list[Any]) -> Callable[[ast.AST], None]:
		return lambda node: primitiveList.append(cast(ast.stmt, node))

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
	def __init__(self
			, findMe: Callable[[ast.AST], bool]
			, doThis: Callable[[ast.AST], ast.AST | Sequence[ast.AST] | None]
			) -> None:
		self.findMe = findMe
		self.doThis = doThis

	def visit(self, node: ast.AST) -> ast.AST | Sequence[ast.AST] | None:
		if self.findMe(node):
			return self.doThis(node)
		return super().visit(node)

def shatter_dataclassesDOTdataclass(dataclass: ast.AST) -> list[ast.AnnAssign]:
	if not isinstance(dataclass, ast.ClassDef):
		return []

	listAnnAssign: list[ast.AnnAssign] = []
	NodeReplacer(ifThis.isAnnAssign(), Then.appendTo(listAnnAssign)).visit(dataclass)
	return listAnnAssign

class UniversalImportTracker:
	def __init__(self, startWith: ast.AST | None = None) -> None:
		self.dictionaryImportFrom: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
		self.listImport: list[str] = []

		if startWith:
			self.walkThis(startWith)

	def addAst(self, astImport_: ast.Import | ast.ImportFrom) -> None:
		if isinstance(astImport_, ast.Import):
			for alias in astImport_.names:
				self.listImport.append(alias.name)
		elif isinstance(astImport_, ast.ImportFrom): # type: ignore
			if astImport_.module is not None:
				for alias in astImport_.names:
					self.dictionaryImportFrom[astImport_.module].append((alias.name, alias.asname))

	def addImportStr(self, module: str) -> None:
		self.listImport.append(module)

	def addImportFromStr(self, module: str, name: str, asname: str | None = None) -> None:
		self.dictionaryImportFrom[module].append((name, asname))

	def makeListAst(self) -> list[ast.ImportFrom | ast.Import]:
		listAstImportFrom: list[ast.ImportFrom] = []
		for module, listOfNameTuples in sorted(self.dictionaryImportFrom.items()):
			listAliases: list[ast.alias] = []
			for name, asname in listOfNameTuples:
				listAliases.append(ast.alias(name=name, asname=asname))
			setAliases = set(listAliases)
			listAliases = sorted(setAliases, key=lambda alias: alias.name)
			listAstImportFrom.append(ast.ImportFrom(module=module, names=listAliases, level=0))

		listAstImport: list[ast.Import] = [ast.Import(names=[ast.alias(name=name, asname=None)]) for name in sorted(set(self.listImport))]
		return listAstImportFrom + listAstImport

	def update(self, *fromTracker: 'UniversalImportTracker') -> None:
		"""
		Update this tracker with imports from one or more other trackers.

		Parameters:
			*fromTracker: One or more UniversalImportTracker objects to merge from.
		"""
		# Merge all import-from dictionaries
		self.dictionaryImportFrom = updateExtendPolishDictionaryLists(self.dictionaryImportFrom, *(tracker.dictionaryImportFrom for tracker in fromTracker), destroyDuplicates=True, reorderLists=True)

		for tracker in fromTracker:
			self.listImport.extend(tracker.listImport)

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
		astCall: ast.AST = self.generic_visit(node)
		if (ifThis.CallAsNameIsIn(self.dictionaryFunctions)(astCall)
		and ifThis.CallDoesNotCallItself("", astCall.func.id)(astCall)): # type: ignore
			inlineDefinition: ast.FunctionDef = self.inlineFunctionBody(astCall.func.id) # type: ignore

			if (inlineDefinition and inlineDefinition.body):
				statementTerminating: ast.stmt = inlineDefinition.body[-1]

				if (isinstance(statementTerminating, ast.Return)
				and statementTerminating.value is not None):
					return self.visit(statementTerminating.value)
				elif isinstance(statementTerminating, ast.Expr):
					return self.visit(statementTerminating.value)
				else:
					return ast.Constant(value=None)
		return astCall

	def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.AST]:
		if (ifThis.CallAsNameIsIn(self.dictionaryFunctions)(node.value)
		and ifThis.CallDoesNotCallItself("", node.value.func.id)(node.value)): # type: ignore
			inlineDefinition: ast.FunctionDef = self.inlineFunctionBody(node.value.func.id) # type: ignore
			return [self.visit(stmt) for stmt in inlineDefinition.body]
		return self.generic_visit(node)
