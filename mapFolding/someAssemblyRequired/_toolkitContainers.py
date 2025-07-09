"""
Map folding AST transformation system: Dataclass decomposition containers and reconstruction logic.

This module provides the structural foundation for the map folding AST transformation system by
implementing container classes that decompose dataclass definitions into their constituent AST
components. Building upon the pattern recognition capabilities established in the foundational layer,
these containers enable the systematic transformation of dataclass-based map folding algorithms
into Numba-compatible implementations.

The decomposition process addresses a fundamental challenge in high-performance computing: Numba's
just-in-time compiler cannot directly process dataclass instances but excels at optimizing
operations on primitive values and tuples. The containers bridge this gap by extracting individual
fields, type annotations, initialization patterns, and reconstruction logic as separate AST nodes
that can be manipulated and recombined for different compilation contexts.

Key decomposition capabilities include field extraction from dataclass definitions into function
parameters, type annotation preservation for static analysis, constructor pattern generation for
different field types, instance reconstruction logic for result packaging, and import dependency
tracking for generated code modules. These components form the building blocks for subsequent
transformation stages that generate specialized modules with embedded constants, eliminated dead
code paths, and optimized execution strategies.

The containers support the complete transformation system from high-level dataclass algorithms
to low-level optimized functions while maintaining semantic equivalence and type safety throughout
the compilation process.
"""

from astToolkit import Be, DOT, identifierDotAttribute, LedgerOfImports, Make, NodeTourist, Then
from collections.abc import Callable
from copy import deepcopy
from hunterMakesPy import raiseIfNone
from mapFolding.someAssemblyRequired import IfThis
from typing import Any, cast
import ast
import dataclasses

dummyAssign = Make.Assign([Make.Name("dummyTarget")], Make.Constant(None))
dummySubscript = Make.Subscript(Make.Name("dummy"), Make.Name("slice"))
dummyTuple = Make.Tuple([Make.Name("dummyElement")])

@dataclasses.dataclass
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

	field2AnnAssign: dict[str, ast.AnnAssign | ast.Assign] = dataclasses.field(default_factory=lambda: dict[str, ast.AnnAssign | ast.Assign]())
	"""Maps field names to their corresponding AST assignment expressions for initialization."""

	Z0Z_field2AnnAssign: dict[str, tuple[ast.AnnAssign | ast.Assign, str]] = dataclasses.field(default_factory=lambda: dict[str, tuple[ast.AnnAssign | ast.Assign, str]]())
	"""Temporary mapping for field assignments with constructor type information."""

	fragments4AssignmentOrParameters: ast.Tuple = dummyTuple
	"""AST tuple used as target for assignment to capture returned field values."""

	imports: LedgerOfImports = dataclasses.field(default_factory=LedgerOfImports)
	"""Import records for the dataclass and its constituent field types."""

	list_argAnnotated4ArgumentsSpecification: list[ast.arg] = dataclasses.field(default_factory=lambda: list[ast.arg]())
	"""Function argument nodes with type annotations for parameter specification."""

	list_keyword_field__field4init: list[ast.keyword] = dataclasses.field(default_factory=lambda: list[ast.keyword]())
	"""Keyword arguments for dataclass initialization using field=field format."""

	listAnnotations: list[ast.expr] = dataclasses.field(default_factory=lambda: list[ast.expr]())
	"""Type annotations for each dataclass field in declaration order."""

	listName4Parameters: list[ast.Name] = dataclasses.field(default_factory=lambda: list[ast.Name]())
	"""Name nodes for each dataclass field used as function parameters."""

	listUnpack: list[ast.AnnAssign] = dataclasses.field(default_factory=lambda: list[ast.AnnAssign]())
	"""Annotated assignment statements to extract individual fields from dataclass instances."""

	map_stateDOTfield2Name: dict[ast.AST, ast.Name] = dataclasses.field(default_factory=lambda: dict[ast.AST, ast.Name]())
	"""Maps dataclass attribute access expressions to field name nodes for find-replace operations."""

	repack: ast.Assign = dummyAssign
	"""AST assignment statement that reconstructs the original dataclass instance from individual fields."""

	signatureReturnAnnotation: ast.Subscript = dummySubscript
	"""Tuple-based return type annotation for functions returning decomposed field values."""

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

	astAnnAssignConstructor: ast.AnnAssign|ast.Assign = dataclasses.field(init=False)
	"""AST assignment statement for field initialization with appropriate constructor."""

	Z0Z_hack: tuple[ast.AnnAssign|ast.Assign, str] = dataclasses.field(init=False)
	"""Temporary tuple containing assignment statement and constructor type information."""

	def __post_init__(self, dataclassesDOTdataclassLogicalPathModule: identifierDotAttribute, dataclassClassDef: ast.ClassDef, dataclassesDOTdataclassInstanceIdentifier: str, field: dataclasses.Field[Any]) -> None:
		"""
		Initialize AST components based on the provided dataclass field.

		This method extracts field attributes and constructs corresponding AST nodes
		for various code generation contexts. It handles special cases for array types,
		scalar types, and complex type annotations, creating appropriate constructor
		calls and import requirements.

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

		self.astAnnotation = raiseIfNone(NodeTourist[ast.AnnAssign, ast.Name | None](
			findThis = Be.AnnAssign.targetIs(IfThis.isNameIdentifier(self.name))
			, doThat = Then.extractIt(cast("Callable[[ast.AnnAssign], ast.Name | None]", DOT.annotation))
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
			if dtype_asnameName.id == 'Array3D':
				axesSubscript = Make.Subscript(Make.Name('tuple'), Make.Tuple([Make.Name('uint8'), Make.Name('uint8'), Make.Name('uint8')]))
			ast_expr = Make.Subscript(Make.Name(annotationType), Make.Tuple([axesSubscript, Make.Subscript(Make.Name('dtype'), dtype_asnameName)]))
			constructor = 'array'
			self.ledger.addImportFrom_asStr(moduleWithLogicalPath, constructor)
			dtypeIdentifier: str = dtype.__name__
			self.ledger.addImportFrom_asStr(moduleWithLogicalPath, dtypeIdentifier, dtype_asnameName.id)
			self.astAnnAssignConstructor = Make.AnnAssign(self.astName, ast_expr, Make.Call(Make.Name(constructor), list_keyword=[Make.keyword('dtype', dtype_asnameName)]))
			self.astAnnAssignConstructor = Make.Assign([self.astName], Make.Call(Make.Name(constructor), list_keyword=[Make.keyword('dtype', dtype_asnameName)]))
			self.Z0Z_hack = (self.astAnnAssignConstructor, 'array')
		elif isinstance(self.astAnnotation, ast.Name):
			self.astAnnAssignConstructor = Make.AnnAssign(self.astName, self.astAnnotation, Make.Call(self.astAnnotation, [Make.Constant(-1)]))
			self.Z0Z_hack = (self.astAnnAssignConstructor, 'scalar')
		elif isinstance(self.astAnnotation, ast.Subscript):
			elementConstructor: str = self.metadata['elementConstructor']
			self.ledger.addImportFrom_asStr(dataclassesDOTdataclassLogicalPathModule, elementConstructor)
			takeTheTuple = deepcopy(self.astAnnotation.slice)
			self.astAnnAssignConstructor = Make.AnnAssign(self.astName, self.astAnnotation, takeTheTuple)
			self.Z0Z_hack = (self.astAnnAssignConstructor, elementConstructor)
		if isinstance(self.astAnnotation, ast.Name):
			self.ledger.addImportFrom_asStr(dataclassesDOTdataclassLogicalPathModule, self.astAnnotation.id)
