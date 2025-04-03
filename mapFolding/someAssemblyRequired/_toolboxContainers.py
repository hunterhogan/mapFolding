"""
Container classes for AST transformations and code synthesis.

This module provides container classes used in the AST transformation process
and code synthesis workflows. It acts as a dependency boundary to prevent
circular imports while providing reusable data structures.
"""
from collections import defaultdict
from collections.abc import Sequence
from mapFolding.someAssemblyRequired import ImaAnnotationType, ast_Identifier, be, Make, parseLogicalPath2astModule, str_nameDOTname
from mapFolding.theSSOT import callableDispatcherHARDCODED, The
from pathlib import Path, PurePosixPath
from typing import Literal
from Z0Z_tools import updateExtendPolishDictionaryLists
import ast
import dataclasses

class LedgerOfImports:
	# TODO When resolving the ledger of imports, remove self-referential imports
	# TODO TypeIgnore :/

	def __init__(self, startWith: ast.AST | None = None) -> None:
		self.dictionaryImportFrom: dict[str_nameDOTname, list[tuple[ast_Identifier, ast_Identifier | None]]] = defaultdict(list)
		self.listImport: list[str_nameDOTname] = []
		if startWith:
			self.walkThis(startWith)

	def addAst(self, astImport____: ast.Import | ast.ImportFrom) -> None:
		assert isinstance(astImport____, (ast.Import, ast.ImportFrom)), f"I received {type(astImport____) = }, but I can only accept {ast.Import} and {ast.ImportFrom}."
		if be.Import(astImport____):
			for alias in astImport____.names:
				self.listImport.append(alias.name)
		elif be.ImportFrom(astImport____):
			# TODO fix the mess created by `None` means '.'. I need a `str_nameDOTname` to replace '.'
			if astImport____.module is None:
				astImport____.module = '.'
			for alias in astImport____.names:
				self.dictionaryImportFrom[astImport____.module].append((alias.name, alias.asname))

	def addImport_asStr(self, moduleWithLogicalPath: str_nameDOTname) -> None:
		self.listImport.append(moduleWithLogicalPath)

	# def addImportFrom_asStr(self, moduleWithLogicalPath: str_nameDOTname, name: ast_Identifier, asname: ast_Identifier | None = None) -> None:
	# 	self.dictionaryImportFrom[moduleWithLogicalPath].append((name, asname))

	def addImportFrom_asStr(self, moduleWithLogicalPath: str_nameDOTname, name: ast_Identifier, asname: ast_Identifier | None = None) -> None:
		if moduleWithLogicalPath not in self.dictionaryImportFrom:
			self.dictionaryImportFrom[moduleWithLogicalPath] = []
		self.dictionaryImportFrom[moduleWithLogicalPath].append((name, asname))

	def removeImportFromModule(self, moduleWithLogicalPath: str_nameDOTname) -> None:
		self.removeImportFrom(moduleWithLogicalPath, None, None)
		"""Remove all imports from a specific module."""

	def removeImportFrom(self, moduleWithLogicalPath: str_nameDOTname, name: ast_Identifier | None, asname: ast_Identifier | None = None) -> None:
		if moduleWithLogicalPath is None:
			raise SyntaxError(f"I received `{moduleWithLogicalPath = }`, but it must be the name of a module.")
		if moduleWithLogicalPath in self.dictionaryImportFrom:
			"""
			name, 			asname				  	Meaning
			ast_Identifier, ast_Identifier			: remove exact matches
			ast_Identifier, None					: remove exact matches
			None, 			ast_Identifier			: remove all matches for asname and if entry_asname is None remove name == ast_Identifier
			None, 			None					: remove all matches for the module
			"""
			if name is None and asname is None:
				# Remove all entries for the module
				self.dictionaryImportFrom.pop(moduleWithLogicalPath)
			else:
				if name is None:
					self.dictionaryImportFrom[moduleWithLogicalPath] = [(entry_name, entry_asname) for entry_name, entry_asname in self.dictionaryImportFrom[moduleWithLogicalPath]
													if not (entry_asname == asname) and not (entry_asname is None and entry_name == asname)]
				else:
					# Remove exact matches for the module
					self.dictionaryImportFrom[moduleWithLogicalPath] = [(entry_name, entry_asname) for entry_name, entry_asname in self.dictionaryImportFrom[moduleWithLogicalPath]
														if not (entry_name == name and entry_asname == asname)]
				if not self.dictionaryImportFrom[moduleWithLogicalPath]:
					self.dictionaryImportFrom.pop(moduleWithLogicalPath)

	def exportListModuleIdentifiers(self) -> list[ast_Identifier]:
		listModuleIdentifiers: list[ast_Identifier] = list(self.dictionaryImportFrom.keys())
		listModuleIdentifiers.extend(self.listImport)
		return sorted(set(listModuleIdentifiers))

	def makeList_ast(self) -> list[ast.ImportFrom | ast.Import]:
		listImportFrom: list[ast.ImportFrom] = []
		for moduleWithLogicalPath, listOfNameTuples in sorted(self.dictionaryImportFrom.items()):
			listOfNameTuples = sorted(list(set(listOfNameTuples)), key=lambda nameTuple: nameTuple[0])
			list_alias: list[ast.alias] = []
			for name, asname in listOfNameTuples:
				list_alias.append(Make.alias(name, asname))
			if list_alias:
				listImportFrom.append(Make.ImportFrom(moduleWithLogicalPath, list_alias))
		list_astImport: list[ast.Import] = [Make.Import(moduleWithLogicalPath) for moduleWithLogicalPath in sorted(set(self.listImport))]
		return listImportFrom + list_astImport

	def update(self, *fromLedger: 'LedgerOfImports') -> None:
		"""Update this ledger with imports from one or more other ledgers.
		Parameters:
			*fromLedger: One or more other `LedgerOfImports` objects from which to merge.
		"""
		self.dictionaryImportFrom = updateExtendPolishDictionaryLists(self.dictionaryImportFrom, *(ledger.dictionaryImportFrom for ledger in fromLedger), destroyDuplicates=True, reorderLists=True)
		for ledger in fromLedger:
			self.listImport.extend(ledger.listImport)

	def walkThis(self, walkThis: ast.AST) -> None:
		for nodeBuffalo in ast.walk(walkThis):
			if isinstance(nodeBuffalo, (ast.Import, ast.ImportFrom)):
				self.addAst(nodeBuffalo)

@dataclasses.dataclass
class IngredientsFunction:
	"""Everything necessary to integrate a function into a module should be here.
	Parameters:
		astFunctionDef: hint `Make.astFunctionDef()`
	"""
	astFunctionDef: ast.FunctionDef
	imports: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)
	type_ignores: list[ast.TypeIgnore] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class IngredientsModule:
	"""Everything necessary to create one _logical_ `ast.Module` should be here.
	Extrinsic qualities should _probably_ be handled externally.

	Parameters:
		ingredientsFunction (None): One or more `IngredientsFunction` that will appended to `listIngredientsFunctions`.
	"""
	ingredientsFunction: dataclasses.InitVar[Sequence[IngredientsFunction] | IngredientsFunction | None] = None

	# init var with an existing module? method to deconstruct an existing module?

	# `body` attribute of `ast.Module`
	"""NOTE
	- Bare statements in `prologue` and `epilogue` are not 'protected' by `if __name__ == '__main__':` so they will be executed merely by loading the module.
	- The dataclass has methods for modifying `prologue`, `epilogue`, and `launcher`.
	- However, `prologue`, `epilogue`, and `launcher` are `ast.Module` (as opposed to `list[ast.stmt]`), so that you may use tools such as `ast.walk` and `ast.NodeVisitor` on the fields.
	"""
	imports: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)
	"""Modify this field using the methods in `LedgerOfImports`."""
	prologue: ast.Module = Make.Module([],[])
	"""Statements after the imports and before the functions in listIngredientsFunctions."""
	listIngredientsFunctions: list[IngredientsFunction] = dataclasses.field(default_factory=list)
	epilogue: ast.Module = Make.Module([],[])
	"""Statements after the functions in listIngredientsFunctions and before `launcher`."""
	launcher: ast.Module = Make.Module([],[])
	"""`if __name__ == '__main__':`"""

	# `ast.TypeIgnore` statements to supplement those in other fields; `type_ignores` is a parameter for `ast.Module` constructor
	supplemental_type_ignores: list[ast.TypeIgnore] = dataclasses.field(default_factory=list)

	def __post_init__(self, ingredientsFunction: Sequence[IngredientsFunction] | IngredientsFunction | None = None) -> None:
		if ingredientsFunction is not None:
			if isinstance(ingredientsFunction, IngredientsFunction):
				self.appendIngredientsFunction(ingredientsFunction)
			else:
				self.appendIngredientsFunction(*ingredientsFunction)

	def _append_astModule(self, self_astModule: ast.Module, astModule: ast.Module | None, statement: Sequence[ast.stmt] | ast.stmt | None, type_ignores: list[ast.TypeIgnore] | None) -> None:
		"""Append one or more statements to `prologue`."""
		list_body: list[ast.stmt] = []
		listTypeIgnore: list[ast.TypeIgnore] = []
		if astModule is not None and be.Module(astModule):
			list_body.extend(astModule.body)
			listTypeIgnore.extend(astModule.type_ignores)
		if type_ignores is not None:
			listTypeIgnore.extend(type_ignores)
		if statement is not None:
			if isinstance(statement, Sequence):
				list_body.extend(statement)
			else:
				list_body.append(statement)
		self_astModule.body.extend(list_body)
		self_astModule.type_ignores.extend(listTypeIgnore)
		ast.fix_missing_locations(self_astModule)

	def appendPrologue(self, astModule: ast.Module | None = None, statement: Sequence[ast.stmt] | ast.stmt | None = None, type_ignores: list[ast.TypeIgnore] | None = None) -> None:
		"""Append one or more statements to `prologue`."""
		self._append_astModule(self.prologue, astModule, statement, type_ignores)

	def appendEpilogue(self, astModule: ast.Module | None = None, statement: Sequence[ast.stmt] | ast.stmt | None = None, type_ignores: list[ast.TypeIgnore] | None = None) -> None:
		"""Append one or more statements to `epilogue`."""
		self._append_astModule(self.epilogue, astModule, statement, type_ignores)

	def appendLauncher(self, astModule: ast.Module | None = None, statement: Sequence[ast.stmt] | ast.stmt | None = None, type_ignores: list[ast.TypeIgnore] | None = None) -> None:
		"""Append one or more statements to `launcher`."""
		self._append_astModule(self.launcher, astModule, statement, type_ignores)

	def appendIngredientsFunction(self, *ingredientsFunction: IngredientsFunction) -> None:
		"""Append one or more `IngredientsFunction`."""
		for allegedIngredientsFunction in ingredientsFunction:
			if isinstance(allegedIngredientsFunction, IngredientsFunction):
				self.listIngredientsFunctions.append(allegedIngredientsFunction)
			else:
				raise ValueError(f"I received `{type(allegedIngredientsFunction) = }`, but I can only accept `{IngredientsFunction}`.")

	def removeImportFromModule(self, moduleWithLogicalPath: str_nameDOTname) -> None:
		self.removeImportFrom(moduleWithLogicalPath, None, None)
		"""Remove all imports from a specific module."""

	def removeImportFrom(self, moduleWithLogicalPath: str_nameDOTname, name: ast_Identifier | None, asname: ast_Identifier | None = None) -> None:
		"""
		This method modifies all `LedgerOfImports` in this `IngredientsModule` and all `IngredientsFunction` in `listIngredientsFunctions`.
		It is not a "blacklist", so the import from could be added after this modification.
		"""
		self.imports.removeImportFrom(moduleWithLogicalPath, name, asname)
		for ingredientsFunction in self.listIngredientsFunctions:
			ingredientsFunction.imports.removeImportFrom(moduleWithLogicalPath, name, asname)

	@property
	def list_astImportImportFrom(self) -> list[ast.Import | ast.ImportFrom]:
		"""List of `ast.Import` and `ast.ImportFrom` statements."""
		sherpaLedger = LedgerOfImports()
		listLedgers: list[LedgerOfImports] = [self.imports]
		for ingredientsFunction in self.listIngredientsFunctions:
			listLedgers.append(ingredientsFunction.imports)
		sherpaLedger.update(*listLedgers)
		return sherpaLedger.makeList_ast()

	@property
	def body(self) -> list[ast.stmt]:
		list_stmt: list[ast.stmt] = []
		list_stmt.extend(self.list_astImportImportFrom)
		list_stmt.extend(self.prologue.body)
		for ingredientsFunction in self.listIngredientsFunctions:
			list_stmt.append(ingredientsFunction.astFunctionDef)
		list_stmt.extend(self.epilogue.body)
		list_stmt.extend(self.launcher.body)
		# TODO `launcher`, if it exists, must start with `if __name__ == '__main__':` and be indented
		return list_stmt

	@property
	def type_ignores(self) -> list[ast.TypeIgnore]:
		listTypeIgnore: list[ast.TypeIgnore] = self.supplemental_type_ignores
		# listTypeIgnore.extend(self.imports.makeListAst())
		listTypeIgnore.extend(self.prologue.type_ignores)
		for ingredientsFunction in self.listIngredientsFunctions:
			listTypeIgnore.extend(ingredientsFunction.type_ignores)
		listTypeIgnore.extend(self.epilogue.type_ignores)
		listTypeIgnore.extend(self.launcher.type_ignores)
		return listTypeIgnore

@dataclasses.dataclass
class RecipeSynthesizeFlow:
	"""Settings for synthesizing flow."""
	# ========================================
	# Source
	# ========================================
	source_astModule = parseLogicalPath2astModule(The.logicalPathModuleSourceAlgorithm)

	# Figure out dynamic flow control to synthesized modules https://github.com/hunterhogan/mapFolding/issues/4
	sourceCallableDispatcher: ast_Identifier = The.sourceCallableDispatcher
	sourceCallableInitialize: ast_Identifier = The.sourceCallableInitialize
	sourceCallableParallel: ast_Identifier = The.sourceCallableParallel
	sourceCallableSequential: ast_Identifier = The.sourceCallableSequential

	sourceDataclassIdentifier: ast_Identifier = The.dataclassIdentifier
	sourceDataclassInstance: ast_Identifier = The.dataclassInstance
	sourceDataclassInstanceTaskDistribution: ast_Identifier = The.dataclassInstanceTaskDistribution
	sourceLogicalPathModuleDataclass: str_nameDOTname = The.logicalPathModuleDataclass

	sourceConcurrencyManagerNamespace = The.sourceConcurrencyManagerNamespace
	sourceConcurrencyManagerIdentifier = The.sourceConcurrencyManagerIdentifier

	# ========================================
	# Logical identifiers (as opposed to physical identifiers)
	# ========================================
	# Package ================================
	packageIdentifier: ast_Identifier | None = The.packageName

	# Qualified logical path ================================
	logicalPathModuleDataclass: str_nameDOTname = sourceLogicalPathModuleDataclass
	logicalPathFlowRoot: ast_Identifier | None = 'syntheticModules'
	""" `logicalPathFlowRoot` likely corresponds to a physical filesystem directory."""

	# Module ================================
	moduleDispatcher: ast_Identifier = 'numbaCount_doTheNeedful'
	moduleInitialize: ast_Identifier = moduleDispatcher
	moduleParallel: ast_Identifier = moduleDispatcher
	moduleSequential: ast_Identifier = moduleDispatcher

	# Function ================================
	callableDispatcher: ast_Identifier = sourceCallableDispatcher
	callableInitialize: ast_Identifier = sourceCallableInitialize
	callableParallel: ast_Identifier = sourceCallableParallel
	callableSequential: ast_Identifier = sourceCallableSequential
	concurrencyManagerNamespace: ast_Identifier = sourceConcurrencyManagerNamespace
	concurrencyManagerIdentifier: ast_Identifier = sourceConcurrencyManagerIdentifier
	dataclassIdentifier: ast_Identifier = sourceDataclassIdentifier

	# Variable ================================
	dataclassInstance: ast_Identifier = sourceDataclassInstance
	dataclassInstanceTaskDistribution: ast_Identifier = sourceDataclassInstanceTaskDistribution

	# ========================================
	# Computed
	# ========================================
	"""
theFormatStrModuleSynthetic = "{packageFlow}Count"
theFormatStrModuleForCallableSynthetic = theFormatStrModuleSynthetic + "_{callableTarget}"
theModuleDispatcherSynthetic: ast_Identifier = theFormatStrModuleForCallableSynthetic.format(packageFlow=packageFlowSynthetic, callableTarget=The.sourceCallableDispatcher)
theLogicalPathModuleDispatcherSynthetic: str = '.'.join([The.packageName, The.moduleOfSyntheticModules, theModuleDispatcherSynthetic])

	"""
	# logicalPathModuleDispatcher: str = '.'.join([Z0Z_flowLogicalPathRoot, moduleDispatcher])
	# ========================================
	# Filesystem (names of physical objects)
	# ========================================
	pathPackage: PurePosixPath | None = PurePosixPath(The.pathPackage)
	fileExtension: str = The.fileExtension

	def _makePathFilename(self, filenameStem: str,
			pathRoot: PurePosixPath | None = None,
			logicalPathINFIX: str_nameDOTname | None = None,
			fileExtension: str | None = None,
			) -> PurePosixPath:
		"""filenameStem: (hint: the name of the logical module)"""
		if pathRoot is None:
			pathRoot = self.pathPackage or PurePosixPath(Path.cwd())
		if logicalPathINFIX:
			whyIsThisStillAThing: list[str] = logicalPathINFIX.split('.')
			pathRoot = pathRoot.joinpath(*whyIsThisStillAThing)
		if fileExtension is None:
			fileExtension = self.fileExtension
		filename: str = filenameStem + fileExtension
		return pathRoot.joinpath(filename)

	@property
	def pathFilenameDispatcher(self) -> PurePosixPath:
		return self._makePathFilename(filenameStem=self.moduleDispatcher, logicalPathINFIX=self.logicalPathFlowRoot)
	@property
	def pathFilenameInitialize(self) -> PurePosixPath:
		return self._makePathFilename(filenameStem=self.moduleInitialize, logicalPathINFIX=self.logicalPathFlowRoot)
	@property
	def pathFilenameParallel(self) -> PurePosixPath:
		return self._makePathFilename(filenameStem=self.moduleParallel, logicalPathINFIX=self.logicalPathFlowRoot)
	@property
	def pathFilenameSequential(self) -> PurePosixPath:
		return self._makePathFilename(filenameStem=self.moduleSequential, logicalPathINFIX=self.logicalPathFlowRoot)

	def __post_init__(self) -> None:
		if ((self.concurrencyManagerIdentifier is not None and self.concurrencyManagerIdentifier != self.sourceConcurrencyManagerIdentifier) # `submit` # type: ignore
			or ((self.concurrencyManagerIdentifier is None) != (self.concurrencyManagerNamespace is None))): # type: ignore
			import warnings
			warnings.warn(f"If your synthesized module is weird, check `{self.concurrencyManagerIdentifier=}` and `{self.concurrencyManagerNamespace=}`. (ChildProcessError? 'Yeah! Children shouldn't be processing stuff, man.')", category=ChildProcessError, stacklevel=2) # pyright: ignore[reportCallIssue, reportArgumentType] Y'all Pynatics need to be less shrill and focus on making code that doesn't need 8000 error categories.

		# self.logicalPathModuleDispatcher!=logicalPathModuleDispatcherHARDCODED or
		if self.callableDispatcher!=callableDispatcherHARDCODED:
			print(f"fyi: `{self.callableDispatcher=}` but\n\t`{callableDispatcherHARDCODED=}`.")

dummyAssign = Make.Assign([Make.Name("dummyTarget")], Make.Constant(None))
dummySubscript = Make.Subscript(Make.Name("dummy"), Make.Name("slice"))
dummyTuple = Make.Tuple([Make.Name("dummyElement")])

@dataclasses.dataclass
class ShatteredDataclass:
	countingVariableAnnotation: ImaAnnotationType
	"""Type annotation for the counting variable extracted from the dataclass."""
	countingVariableName: ast.Name
	"""AST name node representing the counting variable identifier."""
	field2AnnAssign: dict[ast_Identifier, ast.AnnAssign] = dataclasses.field(default_factory=dict)
	"""Maps field names to their corresponding AST call expressions."""
	Z0Z_field2AnnAssign: dict[ast_Identifier, tuple[ast.AnnAssign, str]] = dataclasses.field(default_factory=dict)
	fragments4AssignmentOrParameters: ast.Tuple = dummyTuple
	"""AST tuple used as target for assignment to capture returned fragments."""
	ledger: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)
	"""Import records for the dataclass and its constituent parts."""
	list_argAnnotated4ArgumentsSpecification: list[ast.arg] = dataclasses.field(default_factory=list)
	"""Function argument nodes with annotations for parameter specification."""
	list_keyword_field__field4init: list[ast.keyword] = dataclasses.field(default_factory=list)
	"""Keyword arguments for dataclass initialization with field=field format."""
	listAnnotations: list[ImaAnnotationType] = dataclasses.field(default_factory=list)
	"""Type annotations for each dataclass field."""
	listName4Parameters: list[ast.Name] = dataclasses.field(default_factory=list)
	"""Name nodes for each dataclass field used as function parameters."""
	listUnpack: list[ast.AnnAssign] = dataclasses.field(default_factory=list)
	"""Annotated assignment statements to extract fields from dataclass."""
	map_stateDOTfield2Name: dict[ast.expr, ast.Name] = dataclasses.field(default_factory=dict)
	"""Maps AST expressions to Name nodes for find-replace operations."""
	repack: ast.Assign = dummyAssign
	"""AST assignment statement that reconstructs the original dataclass instance."""
	signatureReturnAnnotation: ast.Subscript = dummySubscript
	"""tuple-based return type annotation for function definitions."""
