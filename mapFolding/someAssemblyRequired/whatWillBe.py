"""
- Settings for synthesizing the modules used by the package (i.e., the flow for numba)
- Settings for synthesizing modules that could be used by the package (e.g., the flow for JAX)
- Therefore, an abstracted system for creating settings for the package
- And with only a little more effort, an abstracted system for creating settings to synthesize arbitrary subsets of modules for arbitrary packages
"""
from mapFolding.someAssemblyRequired.transformationTools import (
	ast_Identifier,
	executeActionUnlessDescendantMatches,
	extractClassDef,
	extractFunctionDef,
	ifThis,
	Make,
	NodeCollector,
	NodeReplacer,
	strDotStrCuzPyStoopid,
	Then,
)
from mapFolding.filesystem import writeStringToHere
from mapFolding.theSSOT import (
	FREAKOUT,
	getDatatypePackage,
	getSourceAlgorithm,
	theDataclassIdentifier,
	theDataclassInstance,
	theDispatcherCallable,
	theFileExtension,
	theFormatStrModuleForCallableSynthetic,
	theFormatStrModuleSynthetic,
	theLogicalPathModuleDataclass,
	theLogicalPathModuleDispatcherSynthetic,
	theModuleDispatcherSynthetic,
	theModuleOfSyntheticModules,
	thePackageName,
	thePathPackage,
	theSourceInitializeCallable,
	theSourceParallelCallable,
	theSourceSequentialCallable,
)
from autoflake import fix_code as autoflake_fix_code
from collections import defaultdict
from collections.abc import Sequence
from inspect import getsource as inspect_getsource
from mapFolding.someAssemblyRequired.ingredientsNumba import parametersNumbaDEFAULT, parametersNumbaSuperJit, parametersNumbaSuperJitParallel, ParametersNumba
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import NamedTuple
from Z0Z_tools import updateExtendPolishDictionaryLists
import ast
import dataclasses

@dataclasses.dataclass
class RecipeSynthesizeFlow:
	"""Settings for synthesizing flow."""
	# TODO consider `IngredientsFlow` or similar
	# ========================================
	# Source
	sourceAlgorithm: ModuleType = getSourceAlgorithm()
	sourcePython: str = inspect_getsource(sourceAlgorithm)
	source_astModule: ast.Module = ast.parse(sourcePython)
	# https://github.com/hunterhogan/mapFolding/issues/4
	sourceDispatcherCallable: str = theDispatcherCallable
	sourceSequentialCallable: str = theSourceSequentialCallable
	sourceDataclassIdentifier: str = theDataclassIdentifier
	# I still hate the OOP paradigm. But I like this dataclass stuff.

	# ========================================
	# Filesystem
	pathPackage: PurePosixPath = PurePosixPath(thePathPackage)
	fileExtension: str = theFileExtension

	# ========================================
	# Logical identifiers
	# meta
	formatStrModuleSynthetic: str = theFormatStrModuleSynthetic
	formatStrModuleForCallableSynthetic: str = theFormatStrModuleForCallableSynthetic

	# Package
	packageName: ast_Identifier = thePackageName

	# Module
	# https://github.com/hunterhogan/mapFolding/issues/4
	Z0Z_flowLogicalPathRoot: str = theModuleOfSyntheticModules
	moduleDispatcher: str = theModuleDispatcherSynthetic
	logicalPathModuleDataclass: str = theLogicalPathModuleDataclass
	# https://github.com/hunterhogan/mapFolding/issues/4
	# `theLogicalPathModuleDispatcherSynthetic` is a problem. It is defined in theSSOT, but it can also be calculated.
	logicalPathModuleDispatcher: str = theLogicalPathModuleDispatcherSynthetic
	dataConverterModule: str = 'dataNamespaceFlattened'

	# Function
	sequentialCallable: str = sourceSequentialCallable
	dataclassIdentifier: str = sourceDataclassIdentifier
	dataConverterCallable: str = 'unpackDataclassPackUp'
	dispatcherCallable: str = sourceDispatcherCallable

	# Variable
	dataclassInstance: str = theDataclassInstance

class LedgerOfImports:
	def __init__(self, startWith: ast.AST | None = None) -> None:
		self.dictionaryImportFrom: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
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
				for alias in astImport_.names:
					self.dictionaryImportFrom[astImport_.module].append((alias.name, alias.asname))

	def addImportStr(self, module: str) -> None:
		self.listImport.append(module)

	def addImportFromStr(self, module: str, name: str, asname: str | None = None) -> None:
		self.dictionaryImportFrom[module].append((name, asname))

	def exportListModuleNames(self) -> list[str]:
		listModuleNames: list[str] = list(self.dictionaryImportFrom.keys())
		listModuleNames.extend(self.listImport)
		return sorted(set(listModuleNames))

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

@dataclasses.dataclass
class Z0Z_IngredientsDataStructure:
	"""Everything necessary to create a data structure should be here."""
	dataclassDef: ast.ClassDef
	imports: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)

@dataclasses.dataclass
class IngredientsFunction:
	"""Everything necessary to integrate a function into a module should be here."""
	FunctionDef: ast.FunctionDef # hint `Make.astFunctionDef`
	imports: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)

@dataclasses.dataclass
class IngredientsModule:
	"""Everything necessary to create one _logical_ `ast.Module` should be here.
	Extrinsic qualities should be handled externally, such as with `RecipeModule`."""
	# If an `ast.Module` had a logical name that would be reasonable, but Python is firmly opposed
	# to a reasonable namespace, therefore, Hunter, you were silly to add a `name` field to this
	# dataclass for building an `ast.Module`.
	# name: ast_Identifier
	# Hey, genius, note that this is dataclasses.InitVar
	ingredientsFunction: dataclasses.InitVar[Sequence[IngredientsFunction] | IngredientsFunction | None] = None

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
		"""Add one or more `IngredientsFunction`. """
		listLedgers: list[LedgerOfImports] = []
		for definition in ingredientsFunction:
			self.functions.append(definition.FunctionDef)
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
class RecipeCountingFunction:
	"""Settings for synthesizing counting functions."""
	ingredients: IngredientsFunction

@dataclasses.dataclass
class RecipeDispatchFunction:
	# A "dispatcher" must receive a dataclass instance and return a dataclass instance.
	# computationStateComplete: ComputationState = dispatcher(computationStateInitialized)
	# The most critical values in the returned dataclass are foldGroups[0:-1] and leavesTotal
	# self.foldsTotal = DatatypeFoldsTotal(self.foldGroups[0:-1].sum() * self.leavesTotal)
	# the function name is required by IngredientsFunction
	ingredients: IngredientsFunction
	logicalPathModuleDataclass: str = theLogicalPathModuleDataclass
	dataclassIdentifier: str = theDataclassIdentifier
	dataclassInstance: str = theDataclassInstance
	Z0Z_unpackDataclass: bool = True
	countDispatcher: bool = True
	# is this the countDispatcher or what is the information for calling the countDispatcher: import or no? callable identifier? parameters? return type?
	# countDispatcher lives in `theLogicalPathModuleDispatcherSynthetic`
	# countDispatcher is named `theDispatcherCallable`
	# post init
	# addImportFromStr(self, module: str, name: str, asname: str | None = None)

@dataclasses.dataclass
class RecipeModule:
	"""How to get one or more logical `ast.Module` on disk as one physical module."""
	# Physical namespace
	filenameStem: str
	fileExtension: str = theFileExtension
	pathPackage: PurePosixPath = PurePosixPath(thePathPackage)

	# Physical and logical namespace
	packageName: ast_Identifier | None= thePackageName
	logicalPathINFIX: ast_Identifier | strDotStrCuzPyStoopid | None = None # module names other than the module itself and the package name

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
		listModules.append(self.filenameStem)
		return '.'.join(listModules)

	@property
	def pathFilename(self):
		""" `PurePosixPath` ensures os-independent formatting of the `dataclass.field` value,
		but you must convert to `Path` to perform filesystem operations."""
		pathRoot: PurePosixPath = self.pathPackage
		filename: str = self.filenameStem + self.fileExtension
		if self.logicalPathINFIX:
			whyIsThisStillAThing: list[str] = self.logicalPathINFIX.split('.')
			pathRoot = pathRoot.joinpath(*whyIsThisStillAThing)
		return pathRoot.joinpath(filename)

	ingredients: IngredientsModule = IngredientsModule()

	@property
	def absoluteImport(self) -> ast.Import:
		return Make.astImport(self._getLogicalPathAbsolute())

	@property
	def absoluteImportFrom(self) -> ast.ImportFrom:
		""" `from . import theModule` """
		logicalPathParent: str = self._getLogicalPathParent() or '.'
		return Make.astImportFrom(logicalPathParent, [Make.astAlias(self.filenameStem)])

	def writeModule(self) -> None:
		astModule = self.ingredients.export()
		ast.fix_missing_locations(astModule)
		pythonSource: str = ast.unparse(astModule)
		if not pythonSource: raise FREAKOUT
		autoflake_additional_imports: list[str] = self.ingredients.imports.exportListModuleNames()
		if self.packageName:
			autoflake_additional_imports.append(self.packageName)
		pythonSource = autoflake_fix_code(pythonSource, autoflake_additional_imports, expand_star_imports=False, remove_all_unused_imports=False, remove_duplicate_keys = False, remove_unused_variables = False,)
		writeStringToHere(pythonSource, self.pathFilename)

numbaFlow: RecipeSynthesizeFlow = RecipeSynthesizeFlow()

# https://github.com/hunterhogan/mapFolding/issues/3
sourceSequentialFunctionDef = extractFunctionDef(numbaFlow.sourceSequentialCallable, numbaFlow.source_astModule)
if sourceSequentialFunctionDef is None: raise FREAKOUT

numbaCountSequential = RecipeCountingFunction(IngredientsFunction(
	FunctionDef=sourceSequentialFunctionDef,
	imports=LedgerOfImports(numbaFlow.source_astModule)
))

numbaDispatcher = RecipeModule(filenameStem=numbaFlow.moduleDispatcher, fileExtension=numbaFlow.fileExtension, pathPackage=numbaFlow.pathPackage,
								packageName=numbaFlow.packageName, logicalPathINFIX=numbaFlow.Z0Z_flowLogicalPathRoot)

class ParametersSynthesizeNumbaCallable(NamedTuple):
	callableTarget: str
	parametersNumba: ParametersNumba | None = None
	inlineCallables: bool = False

listNumbaCallableDispatchees: list[ParametersSynthesizeNumbaCallable] = [
	ParametersSynthesizeNumbaCallable('countParallel', parametersNumbaSuperJitParallel, True),
	ParametersSynthesizeNumbaCallable('countSequential', parametersNumbaSuperJit, True),
	ParametersSynthesizeNumbaCallable('countInitialize', parametersNumbaDEFAULT, True),
]

_datatypeModuleScalar = ''
_decoratorCallable = ''

# if numba
_datatypeModuleScalar = 'numba'
_decoratorCallable = 'jit'
Z0Z_autoflake_additional_imports: list[str] = []
Z0Z_autoflake_additional_imports.append('numba')

def Z0Z_getDatatypeModuleScalar() -> str:
	return _datatypeModuleScalar

def Z0Z_setDatatypeModuleScalar(moduleName: str) -> str:
	global _datatypeModuleScalar
	_datatypeModuleScalar = moduleName
	return _datatypeModuleScalar

def Z0Z_getDecoratorCallable() -> str:
	return _decoratorCallable

def Z0Z_setDecoratorCallable(decoratorName: str) -> str:
	global _decoratorCallable
	_decoratorCallable = decoratorName
	return _decoratorCallable
