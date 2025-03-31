"""
Container classes for AST transformations and code synthesis.

This module provides container classes used in the AST transformation process
and code synthesis workflows. It acts as a dependency boundary to prevent
circular imports while providing reusable data structures.
"""
from collections import defaultdict
from collections.abc import Sequence
from mapFolding.someAssemblyRequired import ast_Identifier, str_nameDOTname, parseLogicalPath2astModule
from mapFolding.theSSOT import raiseIfNoneGitHubIssueNumber3, The
from mapFolding.theSSOT import callableDispatcherHARDCODED
from pathlib import Path, PurePosixPath
from Z0Z_tools import updateExtendPolishDictionaryLists
import ast
import dataclasses

class LedgerOfImports:
	# TODO When resolving the ledger of imports, remove self-referential imports

	def __init__(self, startWith: ast.AST | None = None) -> None:
		self.dictionaryImportFrom: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
		self.listImport: list[str] = []
		if startWith:
			self.walkThis(startWith)

	def addAst(self, astImport_: ast.Import | ast.ImportFrom) -> None:
		assert isinstance(astImport_, (ast.Import, ast.ImportFrom)), f"Expected ast.Import or ast.ImportFrom, got {type(astImport_)}"
		if isinstance(astImport_, ast.Import):
			for alias in astImport_.names:
				self.listImport.append(alias.name)
		else:
			if astImport_.module is not None:
				for alias in astImport_.names:
					self.dictionaryImportFrom[astImport_.module].append((alias.name, alias.asname))

	def addImportAsStr(self, module: str) -> None:
		self.listImport.append(module)

	def addImportFromAsStr(self, module: str, name: str, asname: str | None = None) -> None:
		self.dictionaryImportFrom[module].append((name, asname))

	def exportListModuleIdentifiers(self) -> list[str]:
		listModuleIdentifiers: list[str] = list(self.dictionaryImportFrom.keys())
		listModuleIdentifiers.extend(self.listImport)
		return sorted(set(listModuleIdentifiers))

	def makeListAst(self) -> list[ast.ImportFrom | ast.Import]:
		from mapFolding.someAssemblyRequired import Make
		listAstImportFrom: list[ast.ImportFrom] = []
		for module, listOfNameTuples in sorted(self.dictionaryImportFrom.items()):
			listOfNameTuples = sorted(list(set(listOfNameTuples)), key=lambda nameTuple: nameTuple[0])
			listAlias: list[ast.alias] = []
			for name, asname in listOfNameTuples:
				listAlias.append(Make.Alias(name, asname))
			listAstImportFrom.append(Make.ImportFrom(module, listAlias))
		listAstImport: list[ast.Import] = [Make.Import(name) for name in sorted(set(self.listImport))]
		return listAstImportFrom + listAstImport

	def update(self, *fromLedger: 'LedgerOfImports') -> None:
		"""Update this ledger with imports from one or more other ledgers.
		Parameters:
			*fromLedger: One or more other `LedgerOfImports` objects from which to merge.
		"""
		self.dictionaryImportFrom = updateExtendPolishDictionaryLists(self.dictionaryImportFrom, *(ledger.dictionaryImportFrom for ledger in fromLedger), destroyDuplicates=True, reorderLists=True)
		for ledger in fromLedger:
			self.listImport.extend(ledger.listImport)

	def walkThis(self, walkThis: ast.AST) -> None:
		for smurf in ast.walk(walkThis):
			if isinstance(smurf, (ast.Import, ast.ImportFrom)):
				self.addAst(smurf)

@dataclasses.dataclass
class IngredientsFunction:
	"""Everything necessary to integrate a function into a module should be here."""
	astFunctionDef: ast.FunctionDef # hint `Make.astFunctionDef`
	imports: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)

@dataclasses.dataclass
class IngredientsModule:
	"""Everything necessary to create one _logical_ `ast.Module` should be here.
	Extrinsic qualities should _probably_ be handled externally.

	Parameters:
		ingredientsFunction (None): One or more `IngredientsFunction` that will be deconstructed and appended to `imports` and `functions`.
	"""
	ingredientsFunction: dataclasses.InitVar[Sequence[IngredientsFunction] | IngredientsFunction | None] = None

	# init var with an existing module? method to deconstruct an existing module?

	# `body` attribute of `ast.Module`
	imports: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)
	prologue: list[ast.stmt] = dataclasses.field(default_factory=list)
	functions: list[ast.FunctionDef | ast.stmt] = dataclasses.field(default_factory=list)
	epilogue: list[ast.stmt] = dataclasses.field(default_factory=list)
	launcher: list[ast.stmt] = dataclasses.field(default_factory=list)

	# parameter for `ast.Module` constructor
	type_ignores: list[ast.TypeIgnore] = dataclasses.field(default_factory=list)

	def __post_init__(self, ingredientsFunction: Sequence[IngredientsFunction] | IngredientsFunction | None = None) -> None:
		if ingredientsFunction is not None:
			if isinstance(ingredientsFunction, IngredientsFunction):
				self.addIngredientsFunction(ingredientsFunction)
			else:
				self.addIngredientsFunction(*ingredientsFunction)

	def addIngredientsFunction(self, *ingredientsFunction: IngredientsFunction) -> None:
		"""Add one or more `IngredientsFunction`."""
		listLedgers: list[LedgerOfImports] = []
		for definition in ingredientsFunction:
			self.functions.append(definition.astFunctionDef)
			listLedgers.append(definition.imports)
		self.imports.update(*listLedgers)

	@property
	def body(self) -> list[ast.stmt]:
		list_stmt: list[ast.stmt] = []
		list_stmt.extend(self.imports.makeListAst())
		list_stmt.extend(self.prologue)
		list_stmt.extend(self.functions)
		list_stmt.extend(self.epilogue)
		list_stmt.extend(self.launcher)
		# TODO `launcher`, if it exists, must start with `if __name__ == '__main__':` and be indented
		return list_stmt

@dataclasses.dataclass
class RecipeSynthesizeFlow:
	"""Settings for synthesizing flow."""
	# ========================================
	# Source
	# ========================================
	source_astModule = parseLogicalPath2astModule(The.logicalPathModuleSourceAlgorithm)

	# Figure out dynamic flow control to synthesized modules https://github.com/hunterhogan/mapFolding/issues/4
	sourceCallableDispatcher: str = The.sourceCallableDispatcher
	sourceCallableInitialize: str = The.sourceCallableInitialize
	sourceCallableParallel: str = The.sourceCallableParallel
	sourceCallableSequential: str = The.sourceCallableSequential

	sourceDataclassIdentifier: str = The.dataclassIdentifier
	sourceDataclassInstance: str = The.dataclassInstance
	sourceDataclassInstanceTaskDistribution: str = The.dataclassInstanceTaskDistribution
	sourceLogicalPathModuleDataclass: str_nameDOTname = The.logicalPathModuleDataclass

	sourceConcurrencyManagerNamespace = The.sourceConcurrencyManagerNamespace
	sourceConcurrencyManagerIdentifier = The.sourceConcurrencyManagerIdentifier

	# ========================================
	# Logical identifiers (as opposed to physical identifiers)
	# ========================================
	# Package ================================
	packageIdentifier: ast_Identifier | None = The.packageName

	# Qualified logical path ================================
	logicalPathModuleDataclass: str = sourceLogicalPathModuleDataclass
	logicalPathFlowRoot: str | None = 'syntheticModules'
	""" `logicalPathFlowRoot` likely corresponds to a physical filesystem directory."""

	# Module ================================
	moduleDispatcher: str = 'numbaCount_doTheNeedful'
	moduleInitialize: str = moduleDispatcher
	moduleParallel: str = moduleDispatcher
	moduleSequential: str = moduleDispatcher

	# Function ================================
	callableDispatcher: str = sourceCallableDispatcher
	callableInitialize: str = sourceCallableInitialize
	callableParallel: str = sourceCallableParallel
	callableSequential: str = sourceCallableSequential
	concurrencyManagerNamespace: str = sourceConcurrencyManagerNamespace
	concurrencyManagerIdentifier: str = sourceConcurrencyManagerIdentifier
	dataclassIdentifier: str = sourceDataclassIdentifier

	# Variable ================================
	dataclassInstance: str = sourceDataclassInstance
	dataclassInstanceTaskDistribution: str = sourceDataclassInstanceTaskDistribution

	# ========================================
	# Computed
	# ========================================
	"""
theFormatStrModuleSynthetic = "{packageFlow}Count"
theFormatStrModuleForCallableSynthetic = theFormatStrModuleSynthetic + "_{callableTarget}"
theModuleDispatcherSynthetic: str = theFormatStrModuleForCallableSynthetic.format(packageFlow=packageFlowSynthetic, callableTarget=The.sourceCallableDispatcher)
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

def astModuleToIngredientsFunction(astModule: ast.AST, identifierFunctionDef: ast_Identifier) -> IngredientsFunction:
	from mapFolding.someAssemblyRequired import extractFunctionDef
	astFunctionDef = extractFunctionDef(astModule, identifierFunctionDef)
	if not astFunctionDef: raise raiseIfNoneGitHubIssueNumber3
	return IngredientsFunction(astFunctionDef, LedgerOfImports(astModule))
