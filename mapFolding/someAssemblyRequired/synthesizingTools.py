from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from mapFolding.theSSOT import FREAKOUT, additional_importsHARDCODED, getDatatypeModule, theFileExtension, myPackageNameIs, pathPackage
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
Z0Z_thisCannotBeTheBestWay: TypeAlias = list[ast.Name] | list[ast.Attribute] | list[ast.Subscript] | list[ast.Name | ast.Attribute] | list[ast.Name | ast.Subscript] | list[ast.Attribute | ast.Subscript] | list[ast.Name | ast.Attribute | ast.Subscript]

# NOTE: this is weak
class YouOughtaKnow(NamedTuple):
	callableSynthesized: str
	pathFilenameForMe: Path
	astForCompetentProgrammers: ast.ImportFrom

"""
I suspect I'm only using 1-2% of the potential of `ifThis`, `Then`, and `NodeReplacer`.
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
		return lambda node: (ifThis.isAnnAssign()(node) and isinstance(node.annotation, ast.Name))

	@staticmethod
	def isAnnotationAttribute() -> Callable[[ast.AST], bool]:
		return lambda node: (isinstance(node, ast.AnnAssign) and isinstance(node.annotation, ast.Attribute))

	@staticmethod
	def isAnyAnnotation() -> Callable[[ast.AST], bool]:
		return ifThis.anyOf(ifThis.isAnnotation(), ifThis.isAnnotationAttribute())

	@staticmethod
	def findAnnotationNames() -> Callable[[ast.AST], bool]:
		return lambda node: isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)

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
	def astAnnAssign(target: ast.Name | ast.Attribute | ast.Subscript, annotation: ast.expr, value: ast.expr | None = None
					, **kwargs: Any
					#, **kwargs: **ast._Attributes[int | None],
					) -> ast.AnnAssign:
		""" `simple: int`: uses a clever int-from-boolean to assign the correct value to the `simple` attribute. So, don't add it as a parameter."""
		return ast.AnnAssign(target, annotation, value, simple=int(isinstance(target, ast.Name)))

	@staticmethod
	def astAssign(listTargets: Any, value: ast.expr, type_comment: str | None=None
				, **kwargs: Any
				#, **kwargs: **ast._Attributes[int | None],
				) -> ast.Assign:
		return ast.Assign(targets=cast(list[ast.expr], listTargets), value=value, type_comment=type_comment, **kwargs)

	@staticmethod
	def astArg(identifier: ast_Identifier, annotation: ast.expr | None = None, type_comment: str | None = None
		, **kwargs: Any
		#, **kwargs: **ast._Attributes[int | None],
	) -> ast.arg:
		return ast.arg(identifier, annotation, type_comment, **kwargs)

	@staticmethod
	def astArgumentsSpecification(posonlyargs: list[ast.arg]=[], args: list[ast.arg]=[], vararg: ast.arg|None=None, kwonlyargs: list[ast.arg]=[], kw_defaults: list[ast.expr|None]=[None], kwarg: ast.arg|None=None, defaults: list[ast.expr]=[]) -> ast.arguments:
		return ast.arguments(posonlyargs=posonlyargs, args=args, vararg=vararg, kwonlyargs=kwonlyargs, kw_defaults=kw_defaults, kwarg=kwarg, defaults=defaults)

	@staticmethod
	def astCall(caller: ast.Name | ast.Attribute, args: Sequence[ast.expr] | None = None, list_astKeywords: Sequence[ast.keyword] | None = None) -> ast.Call:
		return ast.Call(func=caller, args=list(args) if args else [], keywords=list(list_astKeywords) if list_astKeywords else [])

	@staticmethod
	def astFunctionDef(name: ast_Identifier, args: ast.arguments=ast.arguments(), body: list[ast.stmt]=[], decorator_list: list[ast.expr]=[], returns: ast.expr|None=None, type_comment: str|None=None, type_params: list[ast.type_param]=[]
		, **kwargs: Any
		#, **kwargs: **ast._Attributes[int | None],
		) -> ast.FunctionDef:
		return ast.FunctionDef(name=name, args=args, body=body, decorator_list=decorator_list, returns=returns, type_comment=type_comment, type_params=type_params, **kwargs)

	@staticmethod
	def astImport(moduleName: ast_Identifier, asname: ast_Identifier | None = None) -> ast.Import:
		return ast.Import(names=[Make.astAlias(moduleName, asname)])

	@staticmethod
	def astImportFrom(moduleName: ast_Identifier, list_astAlias: list[ast.alias]) -> ast.ImportFrom:
		return ast.ImportFrom(module=moduleName, names=list_astAlias, level=0)

	@staticmethod
	def astKeyword(keywordArgument: ast_Identifier, value: ast.expr
					, **kwargs: Any
					#, **kwargs: **ast._Attributes[int | None],
					) -> ast.keyword:
		return ast.keyword(arg=keywordArgument, value=value, **kwargs)

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
	def astTuple(elements: Sequence[ast.expr], ctx: ast.expr_context | None = None
				, **kwargs: Any
				#, **kwargs: **ast._Attributes[int | None],
				) -> ast.Tuple:
		"""Create an AST Tuple node from a list of expressions.

		Parameters:
			elements: List of AST expressions to include in the tuple.
			ctx: Context for the tuple (Load/Store). Defaults to Load context.
		"""
		if ctx is None:
			ctx = ast.Store()
		return ast.Tuple(elts=list(elements), ctx=ctx, **kwargs)

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
		return lambda node: primitiveList.append(node.annotation.id)

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

def shatter_dataclassesDOTdataclass(dataclass: ast.ClassDef, instance_Identifier: ast_Identifier) -> list[ast.AnnAssign]:
	listAnnAssign: list[ast.AnnAssign] = []

	NodeReplacer(ifThis.isAnnAssign()
				, Then.appendAnnAssignOfNameDOTnameTo(instance_Identifier, listAnnAssign)).visit(dataclass)

	return listAnnAssign

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
		self.dictionaryImportFrom = updateExtendPolishDictionaryLists(self.dictionaryImportFrom, *(ledger.dictionaryImportFrom for ledger in fromLedger), destroyDuplicates=True, reorderLists=True)

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
	functions: list[ast.FunctionDef]
	imports: LedgerOfImports
	name: ast_Identifier

	fileExtension: str = theFileExtension
	packageName: ast_Identifier = myPackageNameIs
	Z0Z_logicalPath: ast_Identifier | strDotStrCuzPyStoopid | None = None # module names other than the module itself and the package name
	Z0Z_pathPackage: Path = pathPackage

	def _getLogicalPathParent(self) -> str:
		listModules = [self.packageName]
		if self.Z0Z_logicalPath:
			listModules.extend(list(self.Z0Z_logicalPath))
		return '.'.join(listModules)

	def _getLogicalPathAbsolute(self) -> str:
		return '.'.join([self._getLogicalPathParent(), self.name])

	@property
	def pathFilename(self) -> Path:
		pathRoot: Path = self.Z0Z_pathPackage

		if self.Z0Z_logicalPath:
			pathRoot = pathRoot / self.Z0Z_logicalPath

		return pathRoot / (self.name + self.fileExtension)

	@property
	def absoluteImport(self) -> ast.Import:
		return Make.astImport(self._getLogicalPathAbsolute())

	@property
	def absoluteImportFrom(self) -> ast.ImportFrom:
		return Make.astImportFrom(self._getLogicalPathParent(), [Make.astAlias(self.name)])

	def addFunction(self, ingredientsFunction: IngredientsFunction) -> None:
		"""Add a function to the module and incorporate its imports.

		Parameters:
			ingredientsFunction: Function with its imports to be added to this module.
		"""
		self.functions.append(ingredientsFunction.FunctionDef)
		self.imports.update(ingredientsFunction.imports)

	def addFunctions(self, *ingredientsFunctions: IngredientsFunction) -> None:
		"""Add multiple functions to the module and incorporate their imports.

		Parameters:
			*ingredientsFunctions: One or more functions with their imports to be added.
		"""
		for ingredientsFunction in ingredientsFunctions:
			self.addFunction(ingredientsFunction)

	def removeSelfReferencingImports(self) -> None:
		"""Remove any imports that reference this module itself."""
		moduleFullPath = self._getLogicalPathAbsolute()
		parentPath = self._getLogicalPathParent()

		# Remove any direct imports of this module
		if moduleFullPath in self.imports.listImport:
			self.imports.listImport.remove(moduleFullPath)

		# Remove any imports from this module's parent that import this module
		if parentPath in self.imports.dictionaryImportFrom:
			self.imports.dictionaryImportFrom[parentPath] = [
				alias for alias in self.imports.dictionaryImportFrom[parentPath]
				if alias.name != self.name
			]
			# Clean up empty entries
			if not self.imports.dictionaryImportFrom[parentPath]:
				del self.imports.dictionaryImportFrom[parentPath]

	def writeModule(self) -> None:
		"""Writes the module to disk with proper imports and functions.

		This method creates a proper AST module with imports and function definitions,
		fixes missing locations, unpacks the AST to Python code, applies autoflake
		to clean up imports, and writes the resulting code to the appropriate file.
		"""
		# self.removeSelfReferencingImports()
		listAstImports: list[ast.ImportFrom | ast.Import] = self.imports.makeListAst()
		astModule = Make.astModule(body=cast(list[ast.stmt], listAstImports + self.functions))
		ast.fix_missing_locations(astModule)
		pythonSource: str = ast.unparse(astModule)
		if not pythonSource: raise FREAKOUT
		additional_imports: list[str] = additional_importsHARDCODED
		additional_imports.append(getDatatypeModule())
		pythonSource = autoflake.fix_code(pythonSource, additional_imports)
		self.pathFilename.write_text(pythonSource)

	# TODO create logic (as init or methods) for aggregating/incorporating `IngredientsFunction` objects
	# When resolving the ledger of imports, remove self-referential imports
