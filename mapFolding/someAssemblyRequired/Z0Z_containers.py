"""
Container classes for AST transformations and code synthesis.

This module provides container classes used in the AST transformation process
and code synthesis workflows. It acts as a dependency boundary to prevent
circular imports while providing reusable data structures.
"""
from autoflake import fix_code as autoflake_fix_code
from collections import defaultdict
from collections.abc import Sequence
from importlib import import_module as importlib_import_module
from inspect import getsource as inspect_getsource
from mapFolding.filesystem import writeStringToHere
from mapFolding.someAssemblyRequired import ast_Identifier, Make, nameDOTname
from mapFolding.theSSOT import raiseIfNoneGitHubIssueNumber3, The
from os import PathLike
from pathlib import Path, PurePath, PurePosixPath
from types import ModuleType
from typing import Any
from Z0Z_tools import updateExtendPolishDictionaryLists
import ast
import dataclasses
"""
Semiotic notes:
In the `ast` package, some things that look and feel like a "name" are not `ast.Name` type. The following semiotics are a balance between technical precision and practical usage.

astName: always means `ast.Name`.
Name: uppercase, _should_ be interchangeable with astName, even in camelCase.
Hunter: ^^ did you do that ^^ ? Are you sure? You just fixed some "Name" identifiers that should have been "_name" because the wrong case confused you.
name: lowercase, never means `ast.Name`. In camelCase, I _should_ avoid using it in such a way that it could be confused with "Name", uppercase.
_Identifier: very strongly correlates with the private `ast._Identifier`, which is a `TypeAlias` for `str`.
identifier: lowercase, a general term that includes the above and other Python identifiers.
Identifier: uppercase, without the leading underscore should only appear in camelCase and means "identifier", lowercase.
namespace: lowercase, in dotted-names, such as `pathlib.Path` or `collections.abc`, "namespace" is the part before the dot.
Namespace: uppercase, should only appear in camelCase and means "namespace", lowercase.
"""

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
		listAstImportFrom: list[ast.ImportFrom] = []
		for module, listOfNameTuples in sorted(self.dictionaryImportFrom.items()):
			listOfNameTuples = sorted(list(set(listOfNameTuples)), key=lambda nameTuple: nameTuple[0])
			listAlias: list[ast.alias] = []
			for name, asname in listOfNameTuples:
				listAlias.append(Make.astAlias(name, asname))
			listAstImportFrom.append(Make.astImportFrom(module, listAlias))
		listAstImport: list[ast.Import] = [Make.astImport(name) for name in sorted(set(self.listImport))]
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
	Extrinsic qualities should _probably_ be handled externally."""
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

	def _makeModuleBody(self) -> list[ast.stmt]:
		body: list[ast.stmt] = []
		body.extend(self.imports.makeListAst())
		body.extend(self.prologue)
		body.extend(self.functions)
		body.extend(self.epilogue)
		body.extend(self.launcher)
		# TODO `launcher`, if it exists, must start with `if __name__ == '__main__':` and be indented
		return body

	def export(self) -> ast.Module:
		"""Create a new `ast.Module` from the ingredients."""
		return Make.astModule(self._makeModuleBody(), self.type_ignores)

@dataclasses.dataclass
class RecipeSynthesizeFlow:
	"""Settings for synthesizing flow."""
	# ========================================
	# Source
	# ========================================
	# This is probably too restrictive.
	sourceAlgorithm: ModuleType = importlib_import_module(The.logicalPathModuleSourceAlgorithm)
	sourcePython: str = inspect_getsource(sourceAlgorithm)
	source_astModule: ast.Module = ast.parse(sourcePython)

	# Figure out dynamic flow control to synthesized modules https://github.com/hunterhogan/mapFolding/issues/4
	sourceCallableDispatcher: str = The.sourceCallableDispatcher
	sourceCallableInitialize: str = The.sourceCallableInitialize
	sourceCallableParallel: str = The.sourceCallableParallel
	sourceCallableSequential: str = The.sourceCallableSequential

	sourceDataclassIdentifier: str = The.dataclassIdentifier
	sourceDataclassInstance: str = The.dataclassInstance
	sourceDataclassInstanceTaskDistribution: str = The.dataclassInstanceTaskDistribution
	sourcePathModuleDataclass: str = The.logicalPathModuleDataclass

	sourceConcurrencyManagerNamespace = The.sourceConcurrencyManagerNamespace
	sourceConcurrencyManagerIdentifier = The.sourceConcurrencyManagerIdentifier

	# ========================================
	# Logical identifiers
	# ========================================
	# meta ===================================

	# Package ================================
	packageName: ast_Identifier | None = The.packageName

	# Qualified logical path ================================
	logicalPathModuleDataclass: str = sourcePathModuleDataclass
	Z0Z_flowLogicalPathRoot: str | None = 'syntheticModules'

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
	# Filesystem
	# ========================================
	pathPackage: PurePosixPath | None = PurePosixPath(The.pathPackage)
	fileExtension: str = The.fileExtension

	def _makePathFilename(self, filenameStem: str,
			pathRoot: PurePosixPath | None = None,
			logicalPathINFIX: nameDOTname | None = None,
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
		return self._makePathFilename(filenameStem=self.moduleDispatcher, logicalPathINFIX=self.Z0Z_flowLogicalPathRoot)
	@property
	def pathFilenameInitialize(self) -> PurePosixPath:
		return self._makePathFilename(filenameStem=self.moduleInitialize, logicalPathINFIX=self.Z0Z_flowLogicalPathRoot)
	@property
	def pathFilenameParallel(self) -> PurePosixPath:
		return self._makePathFilename(filenameStem=self.moduleParallel, logicalPathINFIX=self.Z0Z_flowLogicalPathRoot)
	@property
	def pathFilenameSequential(self) -> PurePosixPath:
		return self._makePathFilename(filenameStem=self.moduleSequential, logicalPathINFIX=self.Z0Z_flowLogicalPathRoot)

def write_astModule(ingredients: IngredientsModule, pathFilename: str | PathLike[Any] | PurePath, packageName: ast_Identifier | None = None) -> None:
	astModule = ingredients.export()
	ast.fix_missing_locations(astModule)
	pythonSource: str = ast.unparse(astModule)
	if not pythonSource: raise raiseIfNoneGitHubIssueNumber3
	autoflake_additional_imports: list[str] = ingredients.imports.exportListModuleIdentifiers()
	if packageName:
		autoflake_additional_imports.append(packageName)
	pythonSource = autoflake_fix_code(pythonSource, autoflake_additional_imports, expand_star_imports=False, remove_all_unused_imports=False, remove_duplicate_keys = False, remove_unused_variables = False)
	writeStringToHere(pythonSource, pathFilename)

def astModuleToIngredientsFunction(astModule: ast.Module, identifierFunctionDef: ast_Identifier) -> IngredientsFunction:
	from mapFolding.someAssemblyRequired import Z0Z_extractFunctionDef
	astFunctionDef = Z0Z_extractFunctionDef(astModule, identifierFunctionDef)
	if not astFunctionDef: raise raiseIfNoneGitHubIssueNumber3
	return IngredientsFunction(astFunctionDef, LedgerOfImports(astModule))
