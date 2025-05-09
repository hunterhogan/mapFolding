"""
AST Transformation Tools for Python Code Generation

This module provides tools for manipulating and transforming Python abstract syntax trees
to generate optimized code. It implements a system that:

1. Extracts functions and classes from existing modules.
2. Reshapes and transforms them through AST manipulation.
3. Manages dependencies and imports.
4. Generates optimized code with specialized implementations.

The module is particularly focused on transforming general-purpose Python code into
high-performance implementations, especially through dataclass decomposition and
function inlining for Numba compatibility.

At its core, the module implements a transformation assembly-line where code flows from
readable, maintainable implementations to highly optimized versions while preserving
logical structure and correctness.
"""

from astToolkit import ClassIsAndAttribute
from mapFolding.someAssemblyRequired import (
	DeReConstructField2ast,
	IfThis,
	ShatteredDataclass,
)
from astToolkit import(
	Be,
	extractClassDef,
	IngredientsFunction,
	Make,
	NodeChanger,
	parseLogicalPath2astModule,
	str_nameDOTname,
	Then,
)
from astToolkit.transformationTools import unparseFindReplace
from Z0Z_tools import importLogicalPath2Callable
import ast
import dataclasses

def shatter_dataclassesDOTdataclass(logicalPathModule: str_nameDOTname, dataclass_Identifier: str, instance_Identifier: str) -> ShatteredDataclass:
	"""
	Decompose a dataclass definition into AST components for manipulation and code generation.

	This function breaks down a complete dataclass (like ComputationState) into its constituent
	parts as AST nodes, enabling fine-grained manipulation of its fields for code generation.
	It extracts all field definitions, annotations, and metadata, organizing them into a
	ShatteredDataclass that provides convenient access to AST representations needed for
	different code generation contexts.

	The function identifies a special "counting variable" (marked with 'theCountingIdentifier'
	metadata) which is crucial for map folding algorithms, ensuring it's properly accessible
	in the generated code.

	This decomposition is particularly important when generating optimized code (e.g., for Numba)
	where dataclass instances can't be directly used but their fields need to be individually
	manipulated and passed to computational functions.

	Parameters:
		logicalPathModule: The fully qualified module path containing the dataclass definition.
		dataclass_Identifier: The name of the dataclass to decompose.
		instance_Identifier: The variable name to use for the dataclass instance in generated code.

	Returns:
		shatteredDataclass: A ShatteredDataclass containing AST representations of all dataclass components,
			with imports, field definitions, annotations, and repackaging code.

	Raises:
		ValueError: If the dataclass cannot be found in the specified module or if no counting variable is identified in the dataclass.
	"""
	Official_fieldOrder: list[str] = []
	dictionaryDeReConstruction: dict[str, DeReConstructField2ast] = {}

	dataclassClassDef = extractClassDef(parseLogicalPath2astModule(logicalPathModule), dataclass_Identifier)
	if not isinstance(dataclassClassDef, ast.ClassDef): raise ValueError(f"I could not find `{dataclass_Identifier = }` in `{logicalPathModule = }`.")

	countingVariable = None
	for aField in dataclasses.fields(importLogicalPath2Callable(logicalPathModule, dataclass_Identifier)): # pyright: ignore [reportArgumentType]
		Official_fieldOrder.append(aField.name)
		dictionaryDeReConstruction[aField.name] = DeReConstructField2ast(logicalPathModule, dataclassClassDef, instance_Identifier, aField)
		if aField.metadata.get('theCountingIdentifier', False):
			countingVariable = dictionaryDeReConstruction[aField.name].name

	if countingVariable is None:
		import warnings
		warnings.warn(message=f"I could not find the counting variable in `{dataclass_Identifier = }` in `{logicalPathModule = }`.", category=UserWarning)
		raise Exception

	shatteredDataclass = ShatteredDataclass(
		countingVariableAnnotation=dictionaryDeReConstruction[countingVariable].astAnnotation,
		countingVariableName=dictionaryDeReConstruction[countingVariable].astName,
		field2AnnAssign={dictionaryDeReConstruction[field].name: dictionaryDeReConstruction[field].astAnnAssignConstructor for field in Official_fieldOrder},
		Z0Z_field2AnnAssign={dictionaryDeReConstruction[field].name: dictionaryDeReConstruction[field].Z0Z_hack for field in Official_fieldOrder},
		list_argAnnotated4ArgumentsSpecification=[dictionaryDeReConstruction[field].ast_argAnnotated for field in Official_fieldOrder],
		list_keyword_field__field4init=[dictionaryDeReConstruction[field].ast_keyword_field__field for field in Official_fieldOrder if dictionaryDeReConstruction[field].init],
		listAnnotations=[dictionaryDeReConstruction[field].astAnnotation for field in Official_fieldOrder],
		listName4Parameters=[dictionaryDeReConstruction[field].astName for field in Official_fieldOrder],
		listUnpack=[Make.AnnAssign(dictionaryDeReConstruction[field].astName, dictionaryDeReConstruction[field].astAnnotation, dictionaryDeReConstruction[field].ast_nameDOTname) for field in Official_fieldOrder],
		map_stateDOTfield2Name={dictionaryDeReConstruction[field].ast_nameDOTname: dictionaryDeReConstruction[field].astName for field in Official_fieldOrder},
		)
	shatteredDataclass.fragments4AssignmentOrParameters = Make.Tuple(shatteredDataclass.listName4Parameters, ast.Store())
	shatteredDataclass.repack = Make.Assign([Make.Name(instance_Identifier)], value=Make.Call(Make.Name(dataclass_Identifier), list_keyword=shatteredDataclass.list_keyword_field__field4init))
	shatteredDataclass.signatureReturnAnnotation = Make.Subscript(Make.Name('tuple'), Make.Tuple(shatteredDataclass.listAnnotations))

	shatteredDataclass.imports.update(*(dictionaryDeReConstruction[field].ledger for field in Official_fieldOrder))
	shatteredDataclass.imports.addImportFrom_asStr(logicalPathModule, dataclass_Identifier)

	return shatteredDataclass

def removeDataclassFromFunction(ingredientsTarget: IngredientsFunction, shatteredDataclass: ShatteredDataclass) -> IngredientsFunction:
	ingredientsTarget.astFunctionDef.args = Make.arguments(args=shatteredDataclass.list_argAnnotated4ArgumentsSpecification)
	ingredientsTarget.astFunctionDef.returns = shatteredDataclass.signatureReturnAnnotation
	changeReturnCallable = NodeChanger(Be.Return, Then.replaceWith(Make.Return(shatteredDataclass.fragments4AssignmentOrParameters)))
	changeReturnCallable.visit(ingredientsTarget.astFunctionDef)
	ingredientsTarget.astFunctionDef = unparseFindReplace(ingredientsTarget.astFunctionDef, shatteredDataclass.map_stateDOTfield2Name)
	return ingredientsTarget

def unpackDataclassCallFunctionRepackDataclass(ingredientsCaller: IngredientsFunction, targetCallableIdentifier: str, shatteredDataclass: ShatteredDataclass) -> IngredientsFunction:
	astCallTargetCallable = Make.Call(Make.Name(targetCallableIdentifier), shatteredDataclass.listName4Parameters)
	replaceAssignTargetCallable = NodeChanger(ClassIsAndAttribute.valueIs(ast.Assign, IfThis.isCall_Identifier(targetCallableIdentifier)), Then.replaceWith(Make.Assign([shatteredDataclass.fragments4AssignmentOrParameters], value=astCallTargetCallable)))
	unpack4targetCallable = NodeChanger(ClassIsAndAttribute.valueIs(ast.Assign, IfThis.isCall_Identifier(targetCallableIdentifier)), Then.insertThisAbove(shatteredDataclass.listUnpack))
	repack4targetCallable = NodeChanger(ClassIsAndAttribute.valueIs(ast.Assign, IfThis.isCall_Identifier(targetCallableIdentifier)), Then.insertThisBelow([shatteredDataclass.repack]))
	replaceAssignTargetCallable.visit(ingredientsCaller.astFunctionDef)
	unpack4targetCallable.visit(ingredientsCaller.astFunctionDef)
	repack4targetCallable.visit(ingredientsCaller.astFunctionDef)
	return ingredientsCaller
