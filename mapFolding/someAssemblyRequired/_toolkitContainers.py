"""
AST Container Classes for Python Code Generation and Transformation

This module provides specialized container classes that organize AST nodes, imports, and program structure for code
generation and transformation. These classes form the organizational backbone of the code generation system, enabling:

1. Tracking and managing imports with LedgerOfImports.
2. Packaging function definitions with their dependencies via IngredientsFunction.
3. Structuring complete modules with IngredientsModule.
4. Configuring code synthesis with RecipeSynthesizeFlow.
5. Organizing decomposed dataclass representations with ShatteredDataclass.

Together, these container classes implement a component-based architecture for programmatic generation of
high-performance code. They maintain a clean separation between structure and content, allowing transformations to be
applied systematically while preserving relationships between code elements.

The containers work in conjunction with transformation tools that manipulate the contained AST nodes to implement
specific optimizations and transformations.
"""

from collections.abc import Callable
from astToolkit import ClassIsAndAttribute
from copy import deepcopy
from mapFolding.someAssemblyRequired import ast_Identifier, DOT, IfThis, Make, NodeTourist, parseLogicalPath2astModule, str_nameDOTname, Then, LedgerOfImports
from mapFolding import raiseIfNoneGitHubIssueNumber3, The
from pathlib import Path, PurePosixPath
from typing import Any, cast
import ast
import dataclasses

# Consolidate settings classes through inheritance https://github.com/hunterhogan/mapFolding/issues/15
@dataclasses.dataclass
class RecipeSynthesizeFlow:
	"""
	Configure the generation of new modules, including Numba-accelerated code modules.

	RecipeSynthesizeFlow defines the complete blueprint for transforming an original Python algorithm into an optimized,
	accelerated implementation. It specifies:

	1. Source code locations and identifiers.
	2. Target code locations and identifiers.
	3. Naming conventions for generated modules and functions.
	4. File system paths for output files.
	5. Import relationships between components.

	This configuration class serves as a single source of truth for the code generation process, ensuring consistency
	across all generated artifacts while enabling customization of the transformation assembly line.

	The transformation process uses this configuration to extract functions from the source module, transform them
	according to optimization rules, and output properly structured optimized modules with all necessary imports.
	"""
	# ========================================
	# Source
	source_astModule: ast.Module = parseLogicalPath2astModule(The.logicalPathModuleSourceAlgorithm)
	"""AST of the source algorithm module containing the original implementation."""

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
	moduleDispatcher: ast_Identifier = 'numbaCount'
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

	removeDataclassDispatcher: bool = False
	removeDataclassInitialize: bool = False
	removeDataclassParallel: bool = True
	removeDataclassSequential: bool = True
	# ========================================
	# Computed
	# Figure out dynamic flow control to synthesized modules https://github.com/hunterhogan/mapFolding/issues/4
	# theFormatStrModuleSynthetic = "{packageFlow}Count"
	# theFormatStrModuleForCallableSynthetic = theFormatStrModuleSynthetic + "_{callableTarget}"
	# theModuleDispatcherSynthetic: ast_Identifier = theFormatStrModuleForCallableSynthetic.format(packageFlow=packageFlowSynthetic, callableTarget=The.sourceCallableDispatcher)
	# theLogicalPathModuleDispatcherSynthetic: str = '.'.join([The.packageName, The.moduleOfSyntheticModules, theModuleDispatcherSynthetic])
	# logicalPathModuleDispatcher: str = '.'.join([Z0Z_flowLogicalPathRoot, moduleDispatcher])

	# ========================================
	# Filesystem (names of physical objects)
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

dummyAssign = Make.Assign([Make.Name("dummyTarget")], Make.Constant(None))
dummySubscript = Make.Subscript(Make.Name("dummy"), Make.Name("slice"))
dummyTuple = Make.Tuple([Make.Name("dummyElement")])

# Consolidate settings classes through inheritance https://github.com/hunterhogan/mapFolding/issues/15
@dataclasses.dataclass
class ShatteredDataclass:
	countingVariableAnnotation: ast.expr
	"""Type annotation for the counting variable extracted from the dataclass."""

	countingVariableName: ast.Name
	"""AST name node representing the counting variable identifier."""

	field2AnnAssign: dict[ast_Identifier, ast.AnnAssign | ast.Assign] = dataclasses.field(default_factory=lambda: dict[ast_Identifier, ast.AnnAssign | ast.Assign]())
	"""Maps field names to their corresponding AST call expressions."""

	Z0Z_field2AnnAssign: dict[ast_Identifier, tuple[ast.AnnAssign | ast.Assign, str]] = dataclasses.field(default_factory=lambda: dict[ast_Identifier, tuple[ast.AnnAssign | ast.Assign, str]]())

	fragments4AssignmentOrParameters: ast.Tuple = dummyTuple
	"""AST tuple used as target for assignment to capture returned fragments."""

	imports: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)
	"""Import records for the dataclass and its constituent parts."""

	list_argAnnotated4ArgumentsSpecification: list[ast.arg] = dataclasses.field(default_factory=lambda: list[ast.arg]())
	"""Function argument nodes with annotations for parameter specification."""

	list_keyword_field__field4init: list[ast.keyword] = dataclasses.field(default_factory=lambda: list[ast.keyword]())
	"""Keyword arguments for dataclass initialization with field=field format."""

	listAnnotations: list[ast.expr] = dataclasses.field(default_factory=lambda: list[ast.expr]())
	"""Type annotations for each dataclass field."""

	listName4Parameters: list[ast.Name] = dataclasses.field(default_factory=lambda: list[ast.Name]())
	"""Name nodes for each dataclass field used as function parameters."""

	listUnpack: list[ast.AnnAssign] = dataclasses.field(default_factory=lambda: list[ast.AnnAssign]())
	"""Annotated assignment statements to extract fields from dataclass."""

	map_stateDOTfield2Name: dict[ast.AST, ast.Name] = dataclasses.field(default_factory=lambda: dict[ast.AST, ast.Name]())
	"""Maps AST expressions to Name nodes for find-replace operations."""

	repack: ast.Assign = dummyAssign
	"""AST assignment statement that reconstructs the original dataclass instance."""

	signatureReturnAnnotation: ast.Subscript = dummySubscript
	"""tuple-based return type annotation for function definitions."""

@dataclasses.dataclass
class DeReConstructField2ast:
	"""
	Transform a dataclass field into AST node representations for code generation.

	This class extracts and transforms a dataclass Field object into various AST node
	representations needed for code generation. It handles the conversion of field
	attributes, type annotations, and metadata into AST constructs that can be used
	to reconstruct the field in generated code.

	The class is particularly important for decomposing dataclass fields (like those in
	ComputationState) to enable their use in specialized contexts like Numba-optimized
	functions, where the full dataclass cannot be directly used but its contents need
	to be accessible.

	Each field is processed according to its type and metadata to create appropriate
	variable declarations, type annotations, and initialization code as AST nodes.
	"""
	dataclassesDOTdataclassLogicalPathModule: dataclasses.InitVar[str_nameDOTname]
	dataclassClassDef: dataclasses.InitVar[ast.ClassDef]
	dataclassesDOTdataclassInstance_Identifier: dataclasses.InitVar[ast_Identifier]
	field: dataclasses.InitVar[dataclasses.Field[Any]]

	ledger: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)

	name: ast_Identifier = dataclasses.field(init=False)
	typeBuffalo: type[Any] | str | Any = dataclasses.field(init=False)
	default: Any | None = dataclasses.field(init=False)
	default_factory: Callable[..., Any] | None = dataclasses.field(init=False)
	repr: bool = dataclasses.field(init=False)
	hash: bool | None = dataclasses.field(init=False)
	init: bool = dataclasses.field(init=False)
	compare: bool = dataclasses.field(init=False)
	metadata: dict[Any, Any] = dataclasses.field(init=False)
	kw_only: bool = dataclasses.field(init=False)

	astName: ast.Name = dataclasses.field(init=False)
	ast_keyword_field__field: ast.keyword = dataclasses.field(init=False)
	ast_nameDOTname: ast.Attribute = dataclasses.field(init=False)
	astAnnotation: ast.expr = dataclasses.field(init=False)
	ast_argAnnotated: ast.arg = dataclasses.field(init=False)
	astAnnAssignConstructor: ast.AnnAssign|ast.Assign = dataclasses.field(init=False)
	Z0Z_hack: tuple[ast.AnnAssign|ast.Assign, str] = dataclasses.field(init=False)

	def __post_init__(self, dataclassesDOTdataclassLogicalPathModule: str_nameDOTname, dataclassClassDef: ast.ClassDef, dataclassesDOTdataclassInstance_Identifier: ast_Identifier, field: dataclasses.Field[Any]) -> None:
		self.compare = field.compare
		self.default = field.default if field.default is not dataclasses.MISSING else None
		self.default_factory = field.default_factory if field.default_factory is not dataclasses.MISSING else None
		self.hash = field.hash
		self.init = field.init
		self.kw_only = field.kw_only if field.kw_only is not dataclasses.MISSING else False
		self.metadata = dict(field.metadata)
		self.name = field.name
		self.repr = field.repr
		self.typeBuffalo = field.type

		self.astName = Make.Name(self.name)
		self.ast_keyword_field__field = Make.keyword(self.name, self.astName)
		self.ast_nameDOTname = Make.Attribute(Make.Name(dataclassesDOTdataclassInstance_Identifier), self.name)

		sherpa: ast.expr = NodeTourist( # type: ignore
			findThis=ClassIsAndAttribute.targetIs(ast.AnnAssign, IfThis.isName_Identifier(self.name))
			, doThat=Then.extractIt(DOT.annotation) # type: ignore
			).captureLastMatch(dataclassClassDef)

		if sherpa is None: raise raiseIfNoneGitHubIssueNumber3 # type: ignore
		else: self.astAnnotation = sherpa

		self.ast_argAnnotated = Make.arg(self.name, self.astAnnotation)

		dtype = self.metadata.get('dtype', None)
		if dtype:
			moduleWithLogicalPath: str_nameDOTname = 'numpy'
			annotationType = 'ndarray'
			self.ledger.addImportFrom_asStr(moduleWithLogicalPath, annotationType)
			self.ledger.addImportFrom_asStr(moduleWithLogicalPath, 'dtype')
			axesSubscript = Make.Subscript(Make.Name('tuple'), Make.Name('uint8'))
			dtype_asnameName: ast.Name = cast(ast.Name, self.astAnnotation)
			if dtype_asnameName.id == 'Array3D':
				axesSubscript = Make.Subscript(Make.Name('tuple'), Make.Tuple([Make.Name('uint8'), Make.Name('uint8'), Make.Name('uint8')]))
			ast_expr = Make.Subscript(Make.Name(annotationType), Make.Tuple([axesSubscript, Make.Subscript(Make.Name('dtype'), dtype_asnameName)]))
			constructor = 'array'
			self.ledger.addImportFrom_asStr(moduleWithLogicalPath, constructor)
			dtypeIdentifier: ast_Identifier = dtype.__name__
			self.ledger.addImportFrom_asStr(moduleWithLogicalPath, dtypeIdentifier, dtype_asnameName.id)
			self.astAnnAssignConstructor = Make.AnnAssign(self.astName, ast_expr, Make.Call(Make.Name(constructor), list_keyword=[Make.keyword('dtype', dtype_asnameName)]))
			self.astAnnAssignConstructor = Make.Assign([self.astName], Make.Call(Make.Name(constructor), list_keyword=[Make.keyword('dtype', dtype_asnameName)]))
			self.Z0Z_hack = (self.astAnnAssignConstructor, 'array')
		elif isinstance(self.astAnnotation, ast.Name):
			self.astAnnAssignConstructor = Make.AnnAssign(self.astName, self.astAnnotation, Make.Call(self.astAnnotation, [Make.Constant(-1)]))
			self.Z0Z_hack = (self.astAnnAssignConstructor, 'scalar')
		elif isinstance(self.astAnnotation, ast.Subscript):
			elementConstructor: ast_Identifier = self.metadata['elementConstructor']
			self.ledger.addImportFrom_asStr(dataclassesDOTdataclassLogicalPathModule, elementConstructor)
			takeTheTuple = deepcopy(self.astAnnotation.slice)
			self.astAnnAssignConstructor = Make.AnnAssign(self.astName, self.astAnnotation, takeTheTuple)
			self.Z0Z_hack = (self.astAnnAssignConstructor, elementConstructor)
		if isinstance(self.astAnnotation, ast.Name):
			self.ledger.addImportFrom_asStr(dataclassesDOTdataclassLogicalPathModule, self.astAnnotation.id) # pyright: ignore [reportUnknownArgumentType, reportUnknownMemberType, reportIJustCalledATypeGuardMethod_WTF]
