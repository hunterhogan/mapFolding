"""Decompose state dataclasses and adapt generated field-based functions.

(AI generated docstring)

You can use this module to extract field syntax from Python dataclasses [1], replace
state-object parameters with field parameters, and reconstruct state after generated calls.

Contents
--------
Classes
	DeReConstructField2ast
		Represent one dataclass field as reusable syntax nodes.
	ShatteredDataclass
		Store field syntax for removing and reconstructing a dataclass.
Functions
	findDataclass
		Extract the source module, type, and instance name of a function parameter.
	removeDataclass
		Replace a dataclass parameter with individual field parameters.
	shatterDataclass
		Decompose a source dataclass into field and reconstruction syntax.
	toFieldsToCallToDataclass
		Adapt assigned calls to unpack fields and reconstruct a dataclass.

References
----------
[1] Python dataclass fields and initialization.
	https://docs.python.org/3/library/dataclasses.html
"""
from __future__ import annotations

from astToolkit import Be, DOT, identifierDotAttribute, Make, NodeChanger, NodeTourist, Then
from astToolkit.changeDef import extractClassDef
from astToolkit.containers import IngredientsFunction, LedgerOfImports
from astToolkit.filesystem import parseLogicalPath2astModule
from astToolkit.transformationTools import unparseFindReplace
from copy import deepcopy
from hunterMakesPy import errorL33T, raiseIfNone
from hunterMakesPy.filesystemToolkit import importLogicalPath2Identifier
from mapFolding.kitAST import IfThis
from typing import TYPE_CHECKING
import ast
import dataclasses

if TYPE_CHECKING:
	from collections.abc import Callable
	from typing import Any

# DEVELOPMENT `astToolkit` was created after these dataclasses specifically to make all of this
# easier. That's why they don't use `astToolkit` idioms very often. Try to apply some of these
# concepts https://github.com/python/typing/discussions/2092.

# TODO Figure out how to deconstruct dataclasses with `astToolkit`. It's your effing package, Hunter.
# The downstream code is difficult to change. Hell, the UPSTREAM dataclasses are difficult to change
# because a change can break the ast transformations.
@dataclasses.dataclass(slots=True)
class ShatteredDataclass:
	"""Store field syntax for removing and reconstructing a dataclass.

	(AI generated docstring)

	You can use this container to carry the field declarations, imports, and reconstruction syntax
	produced by `shatterDataclass` [1]. Transformations reuse these nodes to replace a state object
	with individual parameters and a tuple of returned field values.

	Attributes
	----------
	countingVariableAnnotation : ast.expr
		Type annotation for the counting variable extracted from the dataclass.
	countingVariableName : ast.Name
		AST name node representing the counting variable identifier.
	lookupAnnAssignWithConstructor : dict[str, ast.AnnAssign | ast.Assign]
		Maps field names to their corresponding AST assignment expressions for initialization.
	Z0Z_field2AnnAssign : dict[str, tuple[ast.AnnAssign | ast.Assign, str]]
		Field identifiers mapped to initialization syntax and constructor classification.
	fragments4AssignmentOrParameters : ast.Tuple
		Ordered field tuple reused in result expressions and tuple assignment targets.
	imports : LedgerOfImports
		Import records for the dataclass and its constituent field types.
	boxOf_argAnnotated : list[ast.arg]
		Function argument nodes with type annotations for parameter specification.
	boxOf_keyword_field__field4init : list[ast.keyword]
		Keyword arguments for dataclass initialization using field=field format.
	boxOfStaticArrays : list[str]
		Array field identifiers classified as static because their constructor setting is
		`init=False`.
	boxOfStaticScalars : list[str]
		Scalar field identifiers classified as static because their constructor setting is
		`init=False`.
	boxOfAnnotations : list[ast.expr]
		Type annotations for each dataclass field in declaration order.
	boxOfName4Parameters : list[ast.Name]
		Name nodes for each dataclass field used as function parameters.
	boxOfUnpack : list[ast.AnnAssign]
		Annotated assignment statements to extract individual fields from dataclass instances.
	map_stateDOTfield2Name : dict[ast.AST, ast.Name]
		Attribute-access nodes mapped to replacement field-name nodes.
	repack : ast.Assign
		AST assignment statement that reconstructs the original dataclass instance from individual
		fields.
	signatureReturnAnnotation : ast.Subscript
		Tuple-based return type annotation for functions returning decomposed field values.

	References
	----------
	[1] `shatterDataclass`
	"""

	countingVariableAnnotation: ast.expr
	"""Type annotation for the counting variable extracted from the dataclass."""

	countingVariableName: ast.Name
	"""AST name node representing the counting variable identifier."""

	lookupAnnAssignWithConstructor: dict[str, ast.AnnAssign | ast.Assign] = dataclasses.field(default_factory=dict[str, ast.AnnAssign | ast.Assign])
	"""Maps field names to their corresponding AST assignment expressions for initialization."""

	Z0Z_field2AnnAssign: dict[str, tuple[ast.AnnAssign | ast.Assign, str]] = dataclasses.field(default_factory=dict[str, tuple[ast.AnnAssign | ast.Assign, str]])
	"""Temporary mapping for field assignments with constructor type information."""

	fragments4AssignmentOrParameters: ast.Tuple = dataclasses.field(default_factory=lambda: Make.Tuple([Make.Name("dummyElement")]))
	"""AST tuple used as target for assignment to capture returned field values."""

	imports: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)
	"""Import records for the dataclass and its constituent field types."""

	boxOf_argAnnotated: list[ast.arg] = dataclasses.field(default_factory=list[ast.arg])
	"""Function argument nodes with type annotations for parameter specification."""

	boxOf_keyword_field__field4init: list[ast.keyword] = dataclasses.field(default_factory=list[ast.keyword])
	"""Keyword arguments for dataclass initialization using field=field format."""

	boxOfStaticArrays: list[str] = dataclasses.field(default_factory=list[str])
	"""Identifiers of unchanging array fields with `init=False`; mutually exclusive with `boxOf_keyword_field__field4init`."""

	boxOfStaticScalars: list[str] = dataclasses.field(default_factory=list[str])
	"""Identifiers of unchanging scalar fields with `init=False`; mutually exclusive with `boxOf_keyword_field__field4init`."""

	boxOfAnnotations: list[ast.expr] = dataclasses.field(default_factory=list[ast.expr])
	"""Type annotations for each dataclass field in declaration order."""

	boxOfName4Parameters: list[ast.Name] = dataclasses.field(default_factory=list[ast.Name])
	"""Name nodes for each dataclass field used as function parameters."""

	boxOfUnpack: list[ast.AnnAssign] = dataclasses.field(default_factory=list[ast.AnnAssign])
	"""Annotated assignment statements to extract individual fields from dataclass instances."""

	map_stateDOTfield2Name: dict[ast.AST, ast.Name] = dataclasses.field(default_factory=dict[ast.AST, ast.Name])
	"""Maps dataclass attribute access expressions to field name nodes for find-replace operations."""

	repack: ast.Assign = dataclasses.field(default_factory=lambda: Make.Assign([Make.Name("dummyTarget")], Make.Constant(None)))
	"""AST assignment statement that reconstructs the original dataclass instance from individual fields."""

	signatureReturnAnnotation: ast.Subscript = dataclasses.field(default_factory=lambda: Make.Subscript(Make.Name("dummy"), Make.Name("slice")))
	"""Tuple-based return type annotation for functions returning decomposed field values."""

@dataclasses.dataclass(slots=True)
class DeReConstructField2ast:
	"""Represent one dataclass field as reusable syntax nodes.

	(AI generated docstring)

	You can use this container to inspect one source field and obtain declarations, parameters,
	attribute access, and constructor syntax for generated functions. The container copies field
	settings from Python dataclass metadata [1] and retains required imports.

	Construction
	------------
	The generated initializer accepts `dataclassesDOTdataclassLogicalPathModule`, `dataclassClassDef`,
	`dataclassesDOTdataclassInstanceIdentifier`, and `field` as initialization-only values [1].
	The source annotation is read from the assignment for `field.name`.

	Constructor Selection
	---------------------
	Truthy `dtype` metadata selects array construction. A simple name annotation selects scalar
	construction. A subscript annotation selects tuple syntax with an optional `elementConstructor`.

	Attributes
	----------
	ledger : LedgerOfImports
		Import tracking for types and modules required by this field.
	name : str
		Field name extracted from the dataclass field definition.
	typeBuffalo : type[Any] | str | Any
		Type annotation of the field as specified in the dataclass.
	default : Any | None
		Declared field default, or `None` when no default is declared.
	default_factory : Callable[..., Any] | None
		Declared default factory, or `None` when no factory is declared.
	repr : bool
		Whether the field should be included in the string representation.
	hash : bool | None
		Whether the field should be included in hash computation.
	init : bool
		Whether the source field participates in the generated dataclass initializer.
	compare : bool
		Whether the field should be included in comparison operations.
	metadata : dict[Any, Any]
		Field metadata dictionary containing additional configuration information.
	kw_only : bool
		Whether the field must be specified as a keyword-only argument.
	astName : ast.Name
		AST name node representing the field identifier.
	ast_keyword_field__field : ast.keyword
		AST keyword argument for dataclass initialization using field=field pattern.
	ast_nameDOTname : ast.Attribute
		AST attribute access expression for accessing the field from an instance.
	astAnnotation : ast.expr
		Source annotation syntax extracted for the field.
	ast_argAnnotated : ast.arg
		AST function argument with type annotation for parameter specification.
	astAnnAssignConstructor : ast.AnnAssign | ast.Assign
		Field initialization syntax selected from annotation shape and metadata.
	Z0Z_hack : tuple[ast.AnnAssign | ast.Assign, str]
		Initialization syntax paired with its scalar, array, or element-constructor classification.

	References
	----------
	[1] Python dataclass fields and initialization.
		https://docs.python.org/3/library/dataclasses.html
	"""

	dataclassesDOTdataclassLogicalPathModule: dataclasses.InitVar[identifierDotAttribute]
	"""Logical path to the module containing the source dataclass definition."""

	dataclassClassDef: dataclasses.InitVar[ast.ClassDef]
	"""AST class definition node for the source dataclass."""

	dataclassesDOTdataclassInstanceIdentifier: dataclasses.InitVar[str]
	"""Variable identifier for the dataclass instance in generated code."""

	field: dataclasses.InitVar[dataclasses.Field[Any]]
	"""Dataclass field object to be transformed into AST components."""

	ledger: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)
	"""Import tracking for types and modules required by this field."""

	name: str = dataclasses.field(init=False)
	"""Field name extracted from the dataclass field definition."""

	typeBuffalo: type[Any] | str | Any = dataclasses.field(init=False)
	"""Type annotation of the field as specified in the dataclass."""

	default: Any | None = dataclasses.field(init=False)
	"""Default value for the field, or None if no default is specified."""

	default_factory: Callable[..., Any] | None = dataclasses.field(init=False)
	"""Default factory function for the field, or None if not specified."""

	repr: bool = dataclasses.field(init=False)
	"""Whether the field should be included in the string representation."""

	hash: bool | None = dataclasses.field(init=False)
	"""Whether the field should be included in hash computation."""

	init: bool = dataclasses.field(init=False)
	"""Whether the field should be included in the generated __init__ method."""

	compare: bool = dataclasses.field(init=False)
	"""Whether the field should be included in comparison operations."""

	metadata: dict[Any, Any] = dataclasses.field(init=False)
	"""Field metadata dictionary containing additional configuration information."""

	kw_only: bool = dataclasses.field(init=False)
	"""Whether the field must be specified as a keyword-only argument."""

	astName: ast.Name = dataclasses.field(init=False)
	"""AST name node representing the field identifier."""

	ast_keyword_field__field: ast.keyword = dataclasses.field(init=False)
	"""AST keyword argument for dataclass initialization using field=field pattern."""

	ast_nameDOTname: ast.Attribute = dataclasses.field(init=False)
	"""AST attribute access expression for accessing the field from an instance."""

	astAnnotation: ast.expr = dataclasses.field(init=False)
	"""AST expression representing the field's type annotation."""

	ast_argAnnotated: ast.arg = dataclasses.field(init=False)
	"""AST function argument with type annotation for parameter specification."""

	astAnnAssignConstructor: ast.AnnAssign | ast.Assign = dataclasses.field(init=False)
	"""AST assignment statement for field initialization with appropriate constructor."""

	Z0Z_hack: tuple[ast.AnnAssign | ast.Assign, str] = dataclasses.field(init=False)
	"""Temporary tuple containing assignment statement and constructor type information."""

	def __post_init__(self, dataclassesDOTdataclassLogicalPathModule: identifierDotAttribute, dataclassClassDef: ast.ClassDef, dataclassesDOTdataclassInstanceIdentifier: str, field: dataclasses.Field[Any]) -> None:
		"""I use this to prepare field syntax during dataclass decomposition.

		(AI generated docstring)

		This method records settings from `field`, locates the source annotation in
		`dataclassClassDef`,
		and initializes the reusable syntax and import ledger. Array construction uses NumPy [1]
		when `field.metadata` supplies a truthy `dtype`.

		Parameters
		----------
		dataclassesDOTdataclassLogicalPathModule : identifierDotAttribute
			Dotted source module path used to import field types and element constructors.
		dataclassClassDef : ast.ClassDef
			Source class syntax containing an annotated assignment for the field.
		dataclassesDOTdataclassInstanceIdentifier : str
			Instance identifier used in generated attribute access.
		field : dataclasses.Field[Any]
			Dataclass field supplying its name, settings, defaults, and construction metadata.

		References
		----------
		[1] NumPy array construction.
			https://numpy.org/doc/stable/reference/generated/numpy.array.html
		"""
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
		self.ast_nameDOTname = Make.Attribute(Make.Name(dataclassesDOTdataclassInstanceIdentifier), self.name)

		self.astAnnotation = raiseIfNone(NodeTourist[ast.AnnAssign, ast.Name](
			findThis=Be.AnnAssign.targetIs(IfThis.isNameIdentifier(self.name))
			, doThat=Then.extractIt(DOT.annotation)
		).captureLastMatch(dataclassClassDef))

		self.ast_argAnnotated = Make.arg(self.name, self.astAnnotation)

		dtype = self.metadata.get('dtype', None)
		if dtype:
			moduleWithLogicalPath: identifierDotAttribute = 'numpy'
			annotationType = 'ndarray'
			self.ledger.addImportFrom_asStr(moduleWithLogicalPath, annotationType)
			self.ledger.addImportFrom_asStr(moduleWithLogicalPath, 'dtype')
			axesSubscript = Make.Subscript(Make.Name('tuple'), Make.Name('uint8'))
			dtype_asnameName: ast.Name = self.astAnnotation
			if dtype_asnameName.id == '形ArrayTotalLeaves3D':
				axesSubscript = Make.Subscript(Make.Name('tuple'), Make.Tuple([Make.Name('uint8'), Make.Name('uint8'), Make.Name('uint8')]))
			if dtype_asnameName.id == '形ArrayTotalLeaves2D':
				axesSubscript = Make.Subscript(Make.Name('tuple'), Make.Tuple([Make.Name('uint8'), Make.Name('uint8')]))
			ast_expr = Make.Subscript(Make.Name(annotationType), Make.Tuple([axesSubscript, Make.Subscript(Make.Name('dtype'), dtype_asnameName)]))
			constructor = 'array'
			self.ledger.addImportFrom_asStr(moduleWithLogicalPath, constructor)
			dtypeIdentifier: str = dtype.__name__
			self.ledger.addImportFrom_asStr(moduleWithLogicalPath, dtypeIdentifier, dtype_asnameName.id)
			self.astAnnAssignConstructor = Make.AnnAssign(self.astName, ast_expr, Make.Call(Make.Name(constructor), list_keyword=[Make.keyword('dtype', dtype_asnameName)]))
			self.astAnnAssignConstructor = Make.Assign([self.astName], Make.Call(Make.Name(constructor), list_keyword=[Make.keyword('dtype', dtype_asnameName)]))
			self.Z0Z_hack = (self.astAnnAssignConstructor, 'array')
		elif isinstance(self.astAnnotation, ast.Name):
			self.astAnnAssignConstructor = Make.AnnAssign(self.astName, self.astAnnotation, Make.Call(self.astAnnotation, [Make.Constant(-errorL33T)]))
			self.Z0Z_hack = (self.astAnnAssignConstructor, 'scalar')
		elif isinstance(self.astAnnotation, ast.Subscript):
			elementConstructor: str = self.metadata.get('elementConstructor', 'generic')
			if elementConstructor != 'generic':
				self.ledger.addImportFrom_asStr(dataclassesDOTdataclassLogicalPathModule, elementConstructor)
			takeTheTuple = deepcopy(self.astAnnotation.slice)
			self.astAnnAssignConstructor = Make.AnnAssign(self.astName, self.astAnnotation, takeTheTuple)
			self.Z0Z_hack = (self.astAnnAssignConstructor, elementConstructor)
		if isinstance(self.astAnnotation, ast.Name):
			self.ledger.addImportFrom_asStr(dataclassesDOTdataclassLogicalPathModule, self.astAnnotation.id)

def shatterDataclass(logicalPathDataclass: identifierDotAttribute, identifierDataclass: str, identifierDataclassInstance: str) -> ShatteredDataclass:
	"""Decompose a source dataclass into field and reconstruction syntax.

	(AI generated docstring)

	You can use this function to import the dataclass identified by `logicalPathDataclass` and
	`identifierDataclass`, then obtain syntax for each field in declaration order. The result
	supports replacing access through `identifierDataclassInstance` with individual field names.
	The returned container is a `ShatteredDataclass` [1].

	Field Metadata
	--------------
	The last field with truthy `theCountingIdentifier` metadata supplies the counting identifier.
	Array fields use `dtype` metadata, and tuple fields may
	use `elementConstructor`. Fields with `init=False` are excluded from constructor keywords [2].

	Parameters
	----------
	logicalPathDataclass : identifierDotAttribute
		Dotted path to the source module containing the dataclass.
	identifierDataclass : str
		Class identifier to locate in source and import at runtime.
	identifierDataclassInstance : str
		Instance identifier to use in generated attribute access.

	Returns
	-------
	shatteredDataclass : ShatteredDataclass
		Field syntax, import records, counting-field metadata, and reconstruction statements.

	Raises
	------
	ValueError
		If the source class, counting field, or a required field annotation cannot be found.

	Examples
	--------
	The module `mapFolding.kitAST.mapFolding._count` uses this call [3].

		```python
		shatterDataclass(*findDataclass(ingredientsFunction))
		```

	References
	----------
	[1] `ShatteredDataclass`

	[2] Python dataclass fields and initialization.
		https://docs.python.org/3/library/dataclasses.html
	[3] `mapFolding.kitAST.mapFolding._count`
	"""
	Official_fieldOrder: list[str] = []
	dictionaryDeReConstruction: dict[str, DeReConstructField2ast] = {}

	dataclassClassDef: ast.ClassDef | None = extractClassDef(parseLogicalPath2astModule(logicalPathDataclass), identifierDataclass)
	if not dataclassClassDef:
		message: str = f"I could not find `{identifierDataclass = }` in `{logicalPathDataclass = }`."
		raise ValueError(message)

	countingVariable: str | None = None
	for aField in dataclasses.fields(importLogicalPath2Identifier(logicalPathDataclass, identifierDataclass)):
		Official_fieldOrder.append(aField.name)
		dictionaryDeReConstruction[aField.name] = DeReConstructField2ast(logicalPathDataclass, dataclassClassDef, identifierDataclassInstance, aField)
		if aField.metadata.get('theCountingIdentifier', False):
			countingVariable = dictionaryDeReConstruction[aField.name].name
	if countingVariable is None:
		message = f"I could not find the counting variable in `{identifierDataclass = }` in `{logicalPathDataclass = }`."
		raise ValueError(message)

	shatteredDataclass = ShatteredDataclass(
		countingVariableAnnotation=dictionaryDeReConstruction[countingVariable].astAnnotation,
		countingVariableName=dictionaryDeReConstruction[countingVariable].astName,
		lookupAnnAssignWithConstructor={dictionaryDeReConstruction[field].name: dictionaryDeReConstruction[field].astAnnAssignConstructor for field in Official_fieldOrder},
		Z0Z_field2AnnAssign={dictionaryDeReConstruction[field].name: dictionaryDeReConstruction[field].Z0Z_hack for field in Official_fieldOrder},
		boxOf_argAnnotated=[dictionaryDeReConstruction[field].ast_argAnnotated for field in Official_fieldOrder],
		boxOf_keyword_field__field4init=[dictionaryDeReConstruction[field].ast_keyword_field__field for field in Official_fieldOrder if dictionaryDeReConstruction[field].init],
		boxOfStaticArrays=[dictionaryDeReConstruction[field].name for field in Official_fieldOrder if (dictionaryDeReConstruction[field].Z0Z_hack[1] == 'array' and not dictionaryDeReConstruction[field].init)],
		boxOfStaticScalars=[dictionaryDeReConstruction[field].name for field in Official_fieldOrder if (dictionaryDeReConstruction[field].Z0Z_hack[1] == 'scalar' and not dictionaryDeReConstruction[field].init)],
		boxOfAnnotations=[dictionaryDeReConstruction[field].astAnnotation for field in Official_fieldOrder],
		boxOfName4Parameters=[dictionaryDeReConstruction[field].astName for field in Official_fieldOrder],
		boxOfUnpack=[Make.AnnAssign(dictionaryDeReConstruction[field].astName, dictionaryDeReConstruction[field].astAnnotation, dictionaryDeReConstruction[field].ast_nameDOTname) for field in Official_fieldOrder],
		map_stateDOTfield2Name={dictionaryDeReConstruction[field].ast_nameDOTname: dictionaryDeReConstruction[field].astName for field in Official_fieldOrder},
	)
	shatteredDataclass.fragments4AssignmentOrParameters = Make.Tuple(shatteredDataclass.boxOfName4Parameters, ast.Store())
	shatteredDataclass.repack = Make.Assign([Make.Name(identifierDataclassInstance)], value=Make.Call(Make.Name(identifierDataclass), list_keyword=shatteredDataclass.boxOf_keyword_field__field4init))
	shatteredDataclass.signatureReturnAnnotation = Make.Subscript(Make.Name('tuple'), Make.Tuple(shatteredDataclass.boxOfAnnotations))

	shatteredDataclass.imports.update(*(dictionaryDeReConstruction[field].ledger for field in Official_fieldOrder))
	shatteredDataclass.imports.addImportFrom_asStr(logicalPathDataclass, identifierDataclass)

	return shatteredDataclass

def removeDataclass(ingredients: IngredientsFunction, shatteredDataclass: ShatteredDataclass) -> IngredientsFunction:
	"""Replace a dataclass parameter with individual field parameters.

	(AI generated docstring)

	You can use this function to change `ingredients` into a function receiving each field from
	`shatteredDataclass`. The function replaces the complete argument specification and every
	return statement, then rewrites instance attribute access as field names.
	The function updates `ingredients` and returns the original container.

	Return Replacement
	------------------
	Each return becomes the ordered field tuple from `shatteredDataclass`.

	Parameters
	----------
	ingredients : IngredientsFunction
		Function container to transform in place.
	shatteredDataclass : ShatteredDataclass
		Ordered field parameters, return annotation, and attribute replacement mapping.

	Returns
	-------
	ingredients : IngredientsFunction
		The original container holding the transformed function definition.

	Examples
	--------
	The module `mapFolding.kitAST.mapFolding._count` uses this call [1].

		```python
		removeDataclass(ingredientsFunction, shatteredDataclass)
		```

	References
	----------
	[1] `mapFolding.kitAST.mapFolding._count`
	"""
	ingredients.astFunctionDef.args = Make.arguments(list_arg=shatteredDataclass.boxOf_argAnnotated)
	ingredients.astFunctionDef.returns = shatteredDataclass.signatureReturnAnnotation
	NodeChanger(Be.Return, Then.replaceWith(Make.Return(shatteredDataclass.fragments4AssignmentOrParameters))).visit(ingredients.astFunctionDef)
	ingredients.astFunctionDef = unparseFindReplace(ingredients.astFunctionDef, shatteredDataclass.map_stateDOTfield2Name)
	return ingredients

def toFieldsToCallToDataclass(ingredients: IngredientsFunction, identifierCallee: str, shatteredDataclass: ShatteredDataclass) -> IngredientsFunction:
	"""Adapt assigned calls to unpack fields and reconstruct a dataclass.

	(AI generated docstring)

	You can use this function to update a caller in `ingredients` after `identifierCallee` starts
	accepting individual field values. Each matching assignment receives field extraction before
	the call, tuple assignment for the result, and dataclass reconstruction after the call.
	The function updates `ingredients` in place and returns the original container.

	Call Selection
	--------------
	The transformation selects assignments whose value is a direct call to `identifierCallee`.
	The replacement uses the field order and constructor keywords from `shatteredDataclass`.

	Parameters
	----------
	ingredients : IngredientsFunction
		Caller function container to transform in place.
	identifierCallee : str
		Function identifier whose assigned calls receive field extraction and state reconstruction.
	shatteredDataclass : ShatteredDataclass
		Ordered argument names and syntax for extracting and reconstructing state.

	Returns
	-------
	ingredients : IngredientsFunction
		The original caller container with matching assigned calls adapted.

	Examples
	--------
	The module `mapFolding.kitAST.mapFolding._count` uses this call [1].

		```python
		toFieldsToCallToDataclass(unRepackDataclass, 名CallableTarget, shatteredDataclassParallel)
		```

	References
	----------
	[1] `mapFolding.kitAST.mapFolding._count`
	"""
	AssignAndCall: ast.Assign = Make.Assign([shatteredDataclass.fragments4AssignmentOrParameters], value=Make.Call(Make.Name(identifierCallee), shatteredDataclass.boxOfName4Parameters))
	NodeChanger(Be.Assign.valueIs(IfThis.isCallIdentifier(identifierCallee)), Then.replaceWith(AssignAndCall)).visit(ingredients.astFunctionDef)
	NodeChanger(Be.Assign.valueIs(IfThis.isCallIdentifier(identifierCallee)), Then.insertThisAbove(shatteredDataclass.boxOfUnpack)).visit(ingredients.astFunctionDef)
	NodeChanger(Be.Assign.valueIs(IfThis.isCallIdentifier(identifierCallee)), Then.insertThisBelow([shatteredDataclass.repack])).visit(ingredients.astFunctionDef)
	return ingredients

def findDataclass(ingredientsFunction: IngredientsFunction) -> tuple[identifierDotAttribute, str, str]:
	"""Extract the source module, type, and instance name of a function parameter.

	(AI generated docstring)

	You can use this function to identify the state dataclass assumed by a function container.
	The function inspects the last argument annotation, finds the last name inside that annotation,
	and resolves the name through the import ledger. The result also includes the last argument name.

	Parameters
	----------
	ingredientsFunction : IngredientsFunction
		Function container with an annotated state argument and a matching import record.

	Returns
	-------
	dataclassIdentifiers : tuple[identifierDotAttribute, str, str]
		Dotted source module path, class identifier, and instance parameter identifier, in that order.

	Examples
	--------
	The module `mapFolding.kitAST.mapFolding._count` uses this call [1].

		```python
		findDataclass(ingredientsFunction)
		```

	References
	----------
	[1] `mapFolding.kitAST.mapFolding._count`
	"""
	dataclassName: ast.expr = raiseIfNone(NodeTourist[ast.arg, ast.expr](Be.arg, Then.extractIt(DOT.annotation)).captureLastMatch(ingredientsFunction.astFunctionDef))
	identifierDataclass: str = raiseIfNone(NodeTourist[ast.Name, str](Be.Name, Then.extractIt(DOT.id)).captureLastMatch(dataclassName))
	logicalPathDataclass = None
	for moduleWithLogicalPath, boxOfNameTuples in ingredientsFunction.imports._dictionaryImportFrom.items():  # ruff: ignore[private-member-access]
		for nameTuple in boxOfNameTuples:
			if nameTuple[0] == identifierDataclass:
				logicalPathDataclass = moduleWithLogicalPath
				break
		if logicalPathDataclass:
			break
	identifierDataclassInstance: identifierDotAttribute = raiseIfNone(NodeTourist[ast.arg, identifierDotAttribute](Be.arg, Then.extractIt(DOT.arg)).captureLastMatch(ingredientsFunction.astFunctionDef))
	return raiseIfNone(logicalPathDataclass), identifierDataclass, identifierDataclassInstance
