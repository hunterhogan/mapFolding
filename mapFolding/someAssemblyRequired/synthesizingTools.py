from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from mapFolding.theSSOT import FREAKOUT, Z0Z_autoflake_additional_imports, getDatatypePackage, theFileExtension, thePackageName, thePathPackage
from pathlib import Path
from typing import Any, cast, NamedTuple, TypeAlias
from Z0Z_tools import updateExtendPolishDictionaryLists
import ast
import autoflake
import dataclasses

# TODO learn whether libcst can help
import libcst

ast_Identifier: TypeAlias = str
strDotStrCuzPyStoopid: TypeAlias = str
strORlist_ast_type_paramORintORNone: TypeAlias = Any
strORintORNone: TypeAlias = Any
Z0Z_thisCannotBeTheBestWay: TypeAlias = list[ast.Name] | list[ast.Attribute] | list[ast.Subscript] | list[ast.Name | ast.Attribute] | list[ast.Name | ast.Subscript] | list[ast.Attribute | ast.Subscript] | list[ast.Name | ast.Attribute | ast.Subscript]

# NOTE: this is weak
class YouOughtaKnow(NamedTuple):
	callableSynthesized: str
	pathFilenameForMe: Path
	astForCompetentProgrammers: ast.ImportFrom

"""
I suspect I'm only using 1-2% of the potential of `ifThis`, `Make`, `Then`, and `NodeReplacer`.
- nesting or chaining
- idk what `@staticmethod` means or what the alternatives are
- I'm at war with the static type checker, instead of the type checker helping me be more explicit and prevent bugs.
"""

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
	def isUnpackingAnArray(identifier:str) -> Callable[[ast.AST], bool]:
		return lambda node: (isinstance(node, ast.Assign)
						and isinstance(node.targets[0], ast.Name)
						and isinstance(node.value, ast.Subscript)
						and isinstance(node.value.value, ast.Name)
						and node.value.value.id == identifier
						and isinstance(node.value.slice, ast.Attribute)
						)

	@staticmethod
	def isAnnotation_astName() -> Callable[[ast.AST], bool]:
		return lambda node: (ifThis.isAnnAssign()(node) and isinstance(node.annotation, ast.Name)) # type: ignore

class Make:
	@staticmethod
	def copy_astCallKeywords(astCall: ast.Call) -> dict[str, Any]:
		"""Extract keyword parameters from a decorator AST node."""
		dictionaryKeywords: dict[str, Any] = {}
		for keywordItem in astCall.keywords:
			if isinstance(keywordItem.value, ast.Constant) and keywordItem.arg is not None:
				dictionaryKeywords[keywordItem.arg] = keywordItem.value.value
		return dictionaryKeywords

	@staticmethod
	def astAlias(name: ast_Identifier, asname: ast_Identifier | None = None) -> ast.alias:
		return ast.alias(name=name, asname=asname)

	@staticmethod
	def astAnnAssign(target: ast.Name | ast.Attribute | ast.Subscript, annotation: ast.expr, value: ast.expr | None = None, **keywordArguments: int) -> ast.AnnAssign:
		""" `simple: int`: uses a clever int-from-boolean to assign the correct value to the `simple` attribute. So, don't add it as a parameter."""
		return ast.AnnAssign(target, annotation, value, simple=int(isinstance(target, ast.Name)), **keywordArguments)

	@staticmethod
	def astAssign(listTargets: Any, value: ast.expr, **keywordArguments: strORintORNone) -> ast.Assign:
		"""keywordArguments: type_comment:str|None, lineno:int, col_offset:int, end_lineno:int|None, end_col_offset:int|None"""
		return ast.Assign(targets=cast(list[ast.expr], listTargets), value=value, **keywordArguments)

	@staticmethod
	def astArg(identifier: ast_Identifier, annotation: ast.expr | None = None, **keywordArguments: strORintORNone) -> ast.arg:
		"""keywordArguments: type_comment:str|None, lineno:int, col_offset:int, end_lineno:int|None, end_col_offset:int|None"""
		return ast.arg(identifier, annotation, **keywordArguments)

	@staticmethod
	def astArgumentsSpecification(posonlyargs: list[ast.arg]=[], args: list[ast.arg]=[], vararg: ast.arg|None=None, kwonlyargs: list[ast.arg]=[], kw_defaults: list[ast.expr|None]=[None], kwarg: ast.arg|None=None, defaults: list[ast.expr]=[]) -> ast.arguments:
		return ast.arguments(posonlyargs=posonlyargs, args=args, vararg=vararg, kwonlyargs=kwonlyargs, kw_defaults=kw_defaults, kwarg=kwarg, defaults=defaults)

	@staticmethod
	def astCall(caller: ast.Name | ast.Attribute, args: Sequence[ast.expr] | None = None, list_astKeywords: Sequence[ast.keyword] | None = None) -> ast.Call:
		return ast.Call(func=caller, args=list(args) if args else [], keywords=list(list_astKeywords) if list_astKeywords else [])

	@staticmethod
	def astFunctionDef(name: ast_Identifier, args: ast.arguments=ast.arguments(), body: list[ast.stmt]=[], decorator_list: list[ast.expr]=[], returns: ast.expr|None=None, **keywordArguments: strORlist_ast_type_paramORintORNone) -> ast.FunctionDef:
		"""keywordArguments: type_comment:str|None, type_params:list[ast.type_param], lineno:int, col_offset:int, end_lineno:int|None, end_col_offset:int|None"""
		return ast.FunctionDef(name=name, args=args, body=body, decorator_list=decorator_list, returns=returns, **keywordArguments)

	@staticmethod
	def astImport(moduleName: ast_Identifier, asname: ast_Identifier | None = None) -> ast.Import:
		return ast.Import(names=[Make.astAlias(moduleName, asname)])

	@staticmethod
	def astImportFrom(moduleName: ast_Identifier, list_astAlias: list[ast.alias]) -> ast.ImportFrom:
		return ast.ImportFrom(module=moduleName, names=list_astAlias, level=0)

	@staticmethod
	def astKeyword(keywordArgument: ast_Identifier, value: ast.expr, **keywordArguments: int) -> ast.keyword:
		return ast.keyword(arg=keywordArgument, value=value, **keywordArguments)

	@staticmethod
	def astModule(body: list[ast.stmt], type_ignores: list[ast.TypeIgnore] = []) -> ast.Module:
		return ast.Module(body=body, type_ignores=type_ignores)

	@staticmethod
	def astName(identifier: ast_Identifier) -> ast.Name:
		return ast.Name(id=identifier, ctx=ast.Load())

	@staticmethod
	def itDOTname(nameChain: ast.Name | ast.Attribute, dotName: str) -> ast.Attribute:
		return ast.Attribute(value=nameChain, attr=dotName, ctx=ast.Load())

	@staticmethod
	def nameDOTname(identifier: ast_Identifier, *dotName: str) -> ast.Name | ast.Attribute:
		nameDOTname: ast.Name | ast.Attribute = Make.astName(identifier)
		if not dotName:
			return nameDOTname
		for suffix in dotName:
			nameDOTname = Make.itDOTname(nameDOTname, suffix)
		return nameDOTname

	@staticmethod
	def astReturn(value: ast.expr | None = None, **keywordArguments: int) -> ast.Return:
		return ast.Return(value=value, **keywordArguments)

	@staticmethod
	def astTuple(elements: Sequence[ast.expr], context: ast.expr_context | None = None, **keywordArguments: int) -> ast.Tuple:
		"""context: Load/Store/Del"""
		context = context or ast.Load()
		return ast.Tuple(elts=list(elements), ctx=context, **keywordArguments)

class Then:
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
	def Z0Z_appendAnnotationNameTo(primitiveList: list[Any]) -> Callable[[ast.AST], None]:
		return lambda node: primitiveList.append(node.annotation.id) # type: ignore

	@staticmethod
	def replaceWith(astStatement: ast.AST) -> Callable[[ast.AST], ast.stmt]:
		return lambda replaceMe: cast(ast.stmt, astStatement)

	@staticmethod
	def removeThis(astNode: ast.AST) -> None:
		return None

	@staticmethod
	def appendAnnAssignOfNameDOTnameTo(instance_Identifier: ast_Identifier, primitiveList: list[Any]) -> Callable[[ast.AST], None]:
		return lambda node: (
			Then.appendTo(primitiveList)
			(Make.astAnnAssign(node.target # type: ignore
							, node.annotation # type: ignore
							, Make.nameDOTname(instance_Identifier
													, node.target.id)))) # type: ignore

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

class LedgerOfImports:
	def __init__(self, startWith: ast.AST | None = None) -> None:
		self.dictionaryImportFrom: dict[str, list[ast.alias]] = defaultdict(list)
		self.listImport: list[str] = []

		if startWith:
			self.walkThis(startWith)

	def addAst(self, astImport_: ast.Import | ast.ImportFrom) -> None:
		if not isinstance(astImport_, (ast.Import, ast.ImportFrom)): # pyright: ignore[reportUnnecessaryIsInstance]
			raise ValueError(f"Expected ast.Import or ast.ImportFrom, got {type(astImport_)}")
		if isinstance(astImport_, ast.Import):
			for alias in astImport_.names:
				self.listImport.append(alias.name)
		else:
			if astImport_.module is not None:
				self.dictionaryImportFrom[astImport_.module].extend(astImport_.names)

	def addImportStr(self, module: str) -> None:
		self.listImport.append(module)

	def addImportFromStr(self, module: str, name: str, asname: str | None = None) -> None:
		self.dictionaryImportFrom[module].append(Make.astAlias(name, asname))

	def makeListAst(self) -> list[ast.ImportFrom | ast.Import]:
		listAstImportFrom: list[ast.ImportFrom] = []
		for module, list_astAlias in sorted(self.dictionaryImportFrom.items()):
			setAliases = set(list_astAlias)
			sortedAliases = sorted(setAliases, key=lambda alias: alias.name)
			listAstImportFrom.append(Make.astImportFrom(module, sortedAliases))

		listAstImport: list[ast.Import] = [Make.astImport(name) for name in sorted(set(self.listImport))]
		return listAstImportFrom + listAstImport

	def update(self, *fromLedger: 'LedgerOfImports') -> None:
		"""
		Update this ledger with imports from one or more other ledgers.

		Parameters:
			*fromTracker: One or more other `LedgerOfImports` objects from which to merge.
		"""
		self.dictionaryImportFrom = updateExtendPolishDictionaryLists(self.dictionaryImportFrom, *(ledger.dictionaryImportFrom for ledger in fromLedger), destroyDuplicates=True, reorderLists=False)

		for ledger in fromLedger:
			self.listImport.extend(ledger.listImport)

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

@dataclasses.dataclass
class IngredientsFunction:
	"""Everything necessary to integrate a function into a module should be here."""
	FunctionDef: ast.FunctionDef
	imports: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)

@dataclasses.dataclass
class IngredientsModule:
	"""Everything necessary to create a module, including the package context, should be here."""
	name: ast_Identifier

	imports: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)
	prologue: list[ast.stmt] = dataclasses.field(default_factory=list)
	functions: list[ast.FunctionDef | ast.stmt] = dataclasses.field(default_factory=list)
	epilogue: list[ast.stmt] = dataclasses.field(default_factory=list)
	launcher: list[ast.stmt] = dataclasses.field(default_factory=list)

	packageName: ast_Identifier | None= thePackageName
	logicalPathINFIX: ast_Identifier | strDotStrCuzPyStoopid | None = None # module names other than the module itself and the package name
	pathPackage: Path = thePathPackage
	fileExtension: str = theFileExtension
	type_ignores: list[ast.TypeIgnore] = dataclasses.field(default_factory=list)

	def _getLogicalPathParent(self) -> str | None:
		listModules: list[ast_Identifier] = []
		if self.packageName:
			listModules.append(self.packageName)
		if self.logicalPathINFIX:
			listModules.append(self.logicalPathINFIX)
		if listModules:
			return '.'.join(listModules)

	def _getLogicalPathAbsolute(self) -> str:
		listModules: list[ast_Identifier] = []
		logicalPathParent: str | None = self._getLogicalPathParent()
		if logicalPathParent:
			listModules.append(logicalPathParent)
		listModules.append(self.name)
		return '.'.join(listModules)

	@property
	def pathFilename(self) -> Path:
		pathRoot: Path = self.pathPackage
		filename = self.name + self.fileExtension
		if self.logicalPathINFIX:
			whyIsThisStillAThing = self.logicalPathINFIX.split('.')
			pathRoot = pathRoot.joinpath(*whyIsThisStillAThing)
		return pathRoot.joinpath(filename)

	@property
	def absoluteImport(self) -> ast.Import:
		return Make.astImport(self._getLogicalPathAbsolute())

	@property
	def absoluteImportFrom(self) -> ast.ImportFrom:
		""" `from . import theModule` """
		logicalPathParent: str | None = self._getLogicalPathParent()
		if logicalPathParent is None:
			logicalPathParent = '.'
		return Make.astImportFrom(logicalPathParent, [Make.astAlias(self.name)])

	def addFunctions(self, *ingredientsFunction: IngredientsFunction) -> None:
		"""Add one or more `IngredientsFunction`. """
		listLedgers: list[LedgerOfImports] = []
		for definition in ingredientsFunction:
			self.functions.append(definition.FunctionDef)
			listLedgers.append(definition.imports)
		self.imports.update(*listLedgers)

	def _makeModuleBody(self) -> list[ast.stmt]:
		"""Constructs the body of the module, including prologue, functions, epilogue, and launcher."""
		body: list[ast.stmt] = []
		body.extend(self.imports.makeListAst())
		body.extend(self.prologue)
		body.extend(self.functions)
		body.extend(self.epilogue)
		body.extend(self.launcher)
		# TODO `launcher` must start with `if __name__ == '__main__':` and be indented
		return body

	def writeModule(self) -> None:
		"""Writes the module to disk with proper imports and functions.

		This method creates a proper AST module with imports and function definitions,
		fixes missing locations, unpacks the AST to Python code, applies autoflake
		to clean up imports, and writes the resulting code to the appropriate file.
		"""
		astModule = Make.astModule(body=self._makeModuleBody(), type_ignores=self.type_ignores)
		ast.fix_missing_locations(astModule)
		pythonSource: str = ast.unparse(astModule)
		if not pythonSource: raise FREAKOUT
		autoflake_additional_imports: list[str] = []
		if self.packageName:
			autoflake_additional_imports.append(self.packageName)
		# TODO LedgerOfImports method: list of package names. autoflake_additional_imports.extend()
		autoflake_additional_imports.append(getDatatypePackage())
		pythonSource = autoflake.fix_code(pythonSource, autoflake_additional_imports, expand_star_imports=False, remove_all_unused_imports=False, remove_duplicate_keys = False, remove_unused_variables = False,)
		self.pathFilename.write_text(pythonSource)

	# TODO When resolving the ledger of imports, remove self-referential imports

def shatter_dataclassesDOTdataclass(module: ast.Module, dataclass_Identifier: ast_Identifier, instance_Identifier: ast_Identifier
									) -> tuple[ast.Name, LedgerOfImports, list[ast.AnnAssign], list[ast.Name], list[ast.keyword], ast.Tuple]:
	"""
	module: the entire module containing the dataclass, including imports and the dataclass itself.
	dataclass_Identifier: the identifier of the dataclass to be dismantled.
	instance_Identifier: in the synthesized module/function/scope, the identifier that will be used. # So many modifiers and qualifiers that the sentence is confusing.
	"""
	# The ledger needs to be discriminating
	ledgerDataclassAndFragments = LedgerOfImports()
	list_astAnnAssign: list[ast.AnnAssign] = []
	list_astKeywordDataclassFragments: list[ast.keyword] = []
	# e.g., [Make.astKeyword(field.name, Make.astName(field.name)) for field in listDataclassFields if field.init]

	astNameDataclass = Make.astName(dataclass_Identifier)
	dataclass = next((statement for statement in module.body if isinstance(statement, ast.ClassDef) and statement.name == dataclass_Identifier), None)
	if not isinstance(dataclass, ast.ClassDef): raise FREAKOUT

	dataclassDismantler = NodeReplacer(ifThis.isAnnAssign()
									, Then.appendAnnAssignOfNameDOTnameTo(instance_Identifier, list_astAnnAssign))
	dataclassDismantler.visit(dataclass)

	list_astNameDataclassFragments: list[ast.Name] = [Make.astName(astAnnAssign.target.id) for astAnnAssign in list_astAnnAssign if isinstance(astAnnAssign.target, ast.Name)]
	astTupleForAssignTargetsToFragments: ast.Tuple = Make.astTuple(list_astNameDataclassFragments, ast.Store())

	return astNameDataclass, ledgerDataclassAndFragments, list_astAnnAssign, list_astNameDataclassFragments, list_astKeywordDataclassFragments, astTupleForAssignTargetsToFragments
