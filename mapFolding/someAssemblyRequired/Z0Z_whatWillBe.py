from mapFolding.someAssemblyRequired.transformationTools import (
	IngredientsFunction,
	IngredientsModule,
	LedgerOfImports,
	Z0Z_RecipeSynthesizeFlow,
	ast_Identifier,
	Z0Z_executeActionUnlessDescendantMatches,
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
	theDataclassIdentifier,
	theDataclassInstance,
	theFileExtension,
	theLogicalPathModuleDataclass,
	thePackageName,
	thePathPackage,
	theSourceInitializeCallable,
	theSourceParallelCallable,
)
from autoflake import fix_code as autoflake_fix_code
from mapFolding.someAssemblyRequired.ingredientsNumba import parametersNumbaDEFAULT, parametersNumbaSuperJit, parametersNumbaSuperJitParallel, ParametersNumba
from pathlib import Path, PurePosixPath
from typing import NamedTuple, cast
import ast
import dataclasses

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

# This is not a dataclass: this is a function.
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
		return None

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

	ingredients: IngredientsModule = dataclasses.field(default_factory=IngredientsModule)

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

class YouOughtaKnowVESTIGIAL(NamedTuple):
	callableSynthesized: str
	pathFilenameForMe: Path
	astForCompetentProgrammers: ast.ImportFrom

class FunctionInlinerVESTIGIAL(ast.NodeTransformer):
	def __init__(self, dictionaryFunctions: dict[str, ast.FunctionDef]) -> None:
		self.dictionaryFunctions: dict[str, ast.FunctionDef] = dictionaryFunctions

	def inlineFunctionBody(self, callableTargetName: str) -> ast.FunctionDef:
		inlineDefinition: ast.FunctionDef = self.dictionaryFunctions[callableTargetName]
		# Process nested calls within the inlined function
		for astNode in ast.walk(inlineDefinition):
			self.visit(astNode)
		return inlineDefinition

	def visit_Call(self, node: ast.Call):
		astCall = self.generic_visit(node)
		if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryFunctions)(astCall):
			inlineDefinition: ast.FunctionDef = self.inlineFunctionBody(cast(ast.Name, cast(ast.Call, astCall).func).id)

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

	def visit_Expr(self, node: ast.Expr):
		if ifThis.CallDoesNotCallItselfAndNameDOTidIsIn(self.dictionaryFunctions)(node.value):
			inlineDefinition: ast.FunctionDef = self.inlineFunctionBody(cast(ast.Name, cast(ast.Call, node.value).func).id)
			return [self.visit(stmt) for stmt in inlineDefinition.body]
		return self.generic_visit(node)
