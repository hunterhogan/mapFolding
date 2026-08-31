"""
Map folding AST transformation system: Core dataclass decomposition and function optimization tools.

This module implements the essential transformation capabilities that form the operational core of
the map folding AST transformation system. Working with the pattern recognition foundation and
decomposition containers established in the foundational layers, these tools execute the critical
transformations that convert dataclass-based functions into optimized implementations suitable
for Numba just-in-time compilation.

The transformation process addresses the fundamental incompatibility between dataclass-dependent
map folding algorithms and Numba's compilation requirements. While dataclass instances provide
clean, maintainable interfaces for complex mathematical state, Numba cannot directly process
these objects but excels at optimizing operations on primitive values and tuples. The tools
bridge this architectural gap through systematic function signature transformation and calling
convention adaptation.

The three-stage transformation pattern implemented here follows a precise sequence: dataclass
decomposition breaks down dataclass definitions into constituent AST components while extracting
field definitions and type annotations; function transformation converts functions accepting
dataclass parameters to functions accepting individual field parameters with updated signatures
and return types; caller adaptation modifies calling code to unpack dataclass instances, invoke
transformed functions, and repack results back into dataclass instances.

This approach enables seamless integration between high-level dataclass-based interfaces and
low-level optimized implementations, maintaining code clarity while achieving performance gains
through specialized compilation paths essential for computationally intensive map folding research.
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
	"""Container for decomposed dataclass components organized as AST nodes for code generation.

	This class holds the decomposed representation of a dataclass, breaking it down into individual
	AST components that can be manipulated and recombined for different code generation contexts.
	It is particularly essential for transforming dataclass-based algorithms into Numba-compatible
	functions where dataclass instances cannot be directly used.

	The decomposition enables individual field access, type annotation extraction, and parameter
	specification generation while maintaining the structural relationships needed to reconstruct
	equivalent functionality using primitive values and tuples.

	All AST components are organized to support both function parameter specification (unpacking
	dataclass fields into individual parameters) and result reconstruction (packing individual
	values back into dataclass instances).
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
	"""
	Transform a dataclass field into AST node representations for code generation.

	This class extracts and transforms a dataclass Field object into various AST node representations
	needed for code generation. It handles the conversion of field attributes, type annotations, and
	metadata into AST constructs that can be used to reconstruct the field in generated code. The
	class is particularly important for decomposing dataclass fields (like those in ComputationState)
	to enable their use in specialized contexts like Numba-optimized functions, where the full
	dataclass cannot be directly used but its contents need to be accessible.

	Each field is processed according to its type and metadata to create appropriate variable
	declarations, type annotations, and initialization code as AST nodes.
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
		"""
		Initialize AST components based on the provided dataclass field.

		This method extracts field attributes and constructs corresponding AST nodes for various code
		generation contexts. It handles special cases for array types, scalar types, and complex type
		annotations, creating appropriate constructor calls and import requirements.

		Parameters
		----------
		dataclassesDOTdataclassLogicalPathModule : identifierDotAttribute
			Module path containing the dataclass
		dataclassClassDef : ast.ClassDef
			AST class definition for type annotation extraction
		dataclassesDOTdataclassInstanceIdentifier : str
			Instance variable name for attribute access
		field : dataclasses.Field[Any]
			Dataclass field to transform
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
			if dtype_asnameName.id == '形Array3DTotalLeaves':
				axesSubscript = Make.Subscript(Make.Name('tuple'), Make.Tuple([Make.Name('uint8'), Make.Name('uint8'), Make.Name('uint8')]))
			if dtype_asnameName.id == '形Array2DTotalLeaves':
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
	"""Decompose a dataclass definition into AST components for manipulation and code generation.

	(AI generated docstring)

	This function breaks down a complete dataclass (like ComputationState) into its constituent parts
	as AST nodes, enabling fine-grained manipulation of its fields for code generation. It extracts
	all field definitions, annotations, and metadata, organizing them into a ShatteredDataclass that
	provides convenient access to AST representations needed for different code generation contexts.

	The function identifies a special "counting variable" (marked with 'theCountingIdentifier'
	metadata) which is crucial for map folding algorithms, ensuring it's properly accessible in the
	generated code.

	This decomposition is particularly important when generating optimized code (e.g., for Numba)
	where dataclass instances can't be directly used but their fields need to be individually
	manipulated and passed to computational functions.

	Parameters
	----------
	logicalPathDataclass : identifierDotAttribute
		The fully qualified module path containing the dataclass definition.
	identifierDataclass : str
		The name of the dataclass to decompose.
	identifierDataclassInstance : str
		The variable name to use for the dataclass instance in generated code.

	Returns
	-------
	ShatteredDataclass
		A ShatteredDataclass containing AST representations of all dataclass components, with imports,
		field definitions, annotations, and repackaging code.

	Raises
	------
	ValueError
		If the dataclass cannot be found in the specified module or if no counting variable is
		identified in the dataclass.

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
	"""Transform a function that operates on dataclass instances to work with individual field parameters.

	(AI generated docstring)

	This function performs the core transformation required for Numba compatibility by removing dataclass
	dependencies from function signatures and implementations. It modifies the target function to:

	1. Replace the single dataclass parameter with individual field parameters.
	2. Update the return type annotation to return a tuple of field values.
	3. Transform return statements to return the tuple of fields.
	4. Replace all dataclass attribute access with direct field variable access.

	This transformation is essential for creating Numba-compatible functions from dataclass-based
	implementations, as Numba cannot handle dataclass instances directly but can efficiently
	process individual primitive values and tuples.

	Parameters
	----------
	ingredients : IngredientsFunction
		The function definition and its dependencies to be transformed.
	shatteredDataclass : ShatteredDataclass
		The decomposed dataclass components providing AST mappings and transformations.

	Returns
	-------
	IngredientsFunction
		The modified function ingredients with dataclass dependencies removed.

	"""
	ingredients.astFunctionDef.args = Make.arguments(list_arg=shatteredDataclass.boxOf_argAnnotated)
	ingredients.astFunctionDef.returns = shatteredDataclass.signatureReturnAnnotation
	NodeChanger(Be.Return, Then.replaceWith(Make.Return(shatteredDataclass.fragments4AssignmentOrParameters))).visit(ingredients.astFunctionDef)
	ingredients.astFunctionDef = unparseFindReplace(ingredients.astFunctionDef, shatteredDataclass.map_stateDOTfield2Name)
	return ingredients

def toFieldsToCallToDataclass(ingredients: IngredientsFunction, identifierCallee: str, shatteredDataclass: ShatteredDataclass) -> IngredientsFunction:
	"""Transform a caller function to interface with a dataclass-free target function.

	(AI generated docstring)

	This function complements `removeDataclass` by modifying calling code to work with
	the transformed target function. It implements the unpacking and repacking pattern required
	when a dataclass-based caller needs to invoke a function that has been converted to accept
	individual field parameters instead of dataclass instances.

	The transformation creates a three-step pattern around the target function call:
	1. Unpack the dataclass instance into individual field variables.
	2. Call the target function with the unpacked field values.
	3. Repack the returned field values back into a dataclass instance.

	This enables seamless integration between dataclass-based high-level code and optimized
	field-based implementations, maintaining the original interface while enabling performance
	optimizations in the target function.

	Parameters
	----------
	ingredients : IngredientsFunction
		The calling function definition and its dependencies to be transformed.
	identifierCallee : str
		The name of the target function being called.
	shatteredDataclass : ShatteredDataclass
		The decomposed dataclass components providing unpacking and repacking logic.

	Returns
	-------
	IngredientsFunction
		The modified caller function with appropriate unpacking and repacking around the target call.

	"""
	AssignAndCall: ast.Assign = Make.Assign([shatteredDataclass.fragments4AssignmentOrParameters], value=Make.Call(Make.Name(identifierCallee), shatteredDataclass.boxOfName4Parameters))
	NodeChanger(Be.Assign.valueIs(IfThis.isCallIdentifier(identifierCallee)), Then.replaceWith(AssignAndCall)).visit(ingredients.astFunctionDef)
	NodeChanger(Be.Assign.valueIs(IfThis.isCallIdentifier(identifierCallee)), Then.insertThisAbove(shatteredDataclass.boxOfUnpack)).visit(ingredients.astFunctionDef)
	NodeChanger(Be.Assign.valueIs(IfThis.isCallIdentifier(identifierCallee)), Then.insertThisBelow([shatteredDataclass.repack])).visit(ingredients.astFunctionDef)
	return ingredients

def findDataclass(ingredientsFunction: IngredientsFunction) -> tuple[identifierDotAttribute, str, str]:
	"""Dynamically extract information about a `dataclass`: the instance identifier, the identifier, and the logical path module.

	Like many things in the "IngredientsFunction/IngredientsModule" ecosystem, this has specific
	requirements. `ingredientsFunction` must have the dataclass as its first parameter. The
	`LedgerOfImports` in `ingredientsFunction` must have the import information for the dataclass. If
	you are not using `IngredientsFunction`, you can still use this function to get the information
	you want.

	```python
	from astToolkit import astModuleToIngredientsFunction

	tupleInformation = findDataclass(astModuleToIngredientsFunction(astAST, identifier))
	```

	Parameters
	----------
	ingredientsFunction : IngredientsFunction
		Function container with AST and import information.

	Returns
	-------
	logicalPathDataclass : identifierDotAttribute
		Logical path from which the `dataclass` is imported, which might not be the real source of the `dataclass`.
	identifierDataclass : str
		Identifier of the `dataclass`.
	identifierDataclassInstance : str
		Identifier of the `dataclass` instance.
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
