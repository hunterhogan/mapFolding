"""
- Settings for synthesizing the modules used by the package (i.e., the flow for numba)
- Settings for synthesizing modules that could be used by the package (e.g., the flow for JAX)
- Therefore, an abstracted system for creating settings for the package
- And with only a little more effort, an abstracted system for creating settings to synthesize arbitrary subsets of modules for arbitrary packages
"""
from mapFolding.someAssemblyRequired.transformationTools import *
from mapFolding.theSSOT import (
	FREAKOUT,
	getDatatypePackage,
	getSourceAlgorithm,
	theDataclassIdentifierAsStr,
	theDataclassInstanceAsStr,
	theDispatcherCallableAsStr,
	theFileExtension,
	theFormatStrModuleForCallableSynthetic,
	theFormatStrModuleSynthetic,
	theLogicalPathModuleDataclass,
	theLogicalPathModuleDispatcherSynthetic,
	theModuleOfSyntheticModules,
	thePackageName,
	thePathPackage,
	Z0Z_sequentialCallableAsStr,
)
from autoflake import fix_code as autoflake_fix_code
from collections import defaultdict
from collections.abc import Sequence
from inspect import getsource as inspect_getsource
from mapFolding.someAssemblyRequired.ingredientsNumba import parametersNumbaDEFAULT, parametersNumbaSuperJit, parametersNumbaSuperJitParallel, ParametersNumba
from pathlib import Path
from types import ModuleType
from typing import NamedTuple
from Z0Z_tools import updateExtendPolishDictionaryLists
import ast
import dataclasses

"""
Start with what is: theDao.py
Create settings that can transform into what I or the user want it to be.

The simplest flow with numba is:
1. one module
2. dispatcher
	- initialize data with makeJob
	- smash dataclass
	- call countSequential
3. countSequential
	- jitted, not super-jitted
	- functions inlined (or I'd have to jit them)
	- return groupsOfFolds
4. recycle the dataclass with groupsOfFolds
5. return the dataclass
"""

@dataclasses.dataclass
class RecipeSynthesizeFlow:
	"""Settings for synthesizing flow."""
	# TODO consider `IngredientsFlow` or similar
	sourceAlgorithm: ModuleType = getSourceAlgorithm()
	sourcePython: str = inspect_getsource(sourceAlgorithm)
	# sourcePython: str = inspect_getsource(self.sourceAlgorithm)
	# "self" is not defined
	# I still hate the OOP paradigm. But I like this dataclass stuff.
	source_astModule: ast.Module = ast.parse(sourcePython)

	# ========================================
	# Filesystem
	pathPackage: Path = thePathPackage
	fileExtension: str = theFileExtension

	# ========================================
	# Logical identifiers
	# meta
	formatStrModuleSynthetic: str = theFormatStrModuleSynthetic
	formatStrModuleForCallableSynthetic: str = theFormatStrModuleForCallableSynthetic

	# Package
	packageName: ast_Identifier = thePackageName

	# Module
	moduleOfSyntheticModules: str = theModuleOfSyntheticModules
	logicalPathModuleDataclass: str = theLogicalPathModuleDataclass
	logicalPathModuleDispatcher: str = theLogicalPathModuleDispatcherSynthetic
	dataConverterModule: str = 'dataNamespaceFlattened'

	# Function
	dataclassIdentifierAsStr: str = theDataclassIdentifierAsStr
	dispatcherCallableAsStr: str = theDispatcherCallableAsStr
	dataConverterCallableAsStr: str = 'flattenData'
	sequentialCallableAsStr: str = Z0Z_sequentialCallableAsStr

	# Variable
	dataclassInstanceAsStr: str = theDataclassInstanceAsStr

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
class IngredientsFunction:
	"""Everything necessary to integrate a function into a module should be here."""
	FunctionDef: ast.FunctionDef # hint `Make.astFunctionDef`
	imports: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)

@dataclasses.dataclass
class IngredientsModule:
	"""Everything necessary to create a module, including the package context, should be here."""
	name: ast_Identifier
	ingredientsFunction: dataclasses.InitVar[Sequence[IngredientsFunction] | IngredientsFunction | None] = None

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
		pythonSource = autoflake_fix_code(pythonSource, autoflake_additional_imports, expand_star_imports=False, remove_all_unused_imports=False, remove_duplicate_keys = False, remove_unused_variables = False,)
		self.pathFilename.write_text(pythonSource)

@dataclasses.dataclass
class RecipeSynthesizeCountingFunction:
	"""Settings for synthesizing counting functions."""
	ingredients: IngredientsFunction

numbaFlow: RecipeSynthesizeFlow = RecipeSynthesizeFlow()

# https://github.com/hunterhogan/mapFolding/issues/3
sequentialFunctionDef = extractFunctionDef(numbaFlow.sequentialCallableAsStr, numbaFlow.source_astModule)
if sequentialFunctionDef is None: raise FREAKOUT

numbaCountSequential = RecipeSynthesizeCountingFunction(IngredientsFunction(
	FunctionDef=sequentialFunctionDef,
	imports=LedgerOfImports(numbaFlow.source_astModule)
))

# the data converter and the dispatcher could be in the same module.

Z0Z_autoflake_additional_imports: list[str] = []
Z0Z_autoflake_additional_imports.append(thePackageName)

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
