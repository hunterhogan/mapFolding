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

from astToolkit import (
	Be, extractClassDef, identifierDotAttribute, IngredientsFunction, Make, NodeChanger, parseLogicalPath2astModule, Then)
from astToolkit.transformationTools import unparseFindReplace
from hunterMakesPy import importLogicalPath2Identifier
from mapFolding.someAssemblyRequired import DeReConstructField2ast, IfThis, ShatteredDataclass
import ast
import dataclasses

def shatter_dataclassesDOTdataclass(logicalPathModule: identifierDotAttribute, dataclassIdentifier: str, instanceIdentifier: str) -> ShatteredDataclass:
	"""Decompose a dataclass definition into AST components for manipulation and code generation.

	(AI generated docstring)

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

	Parameters
	----------
	logicalPathModule : identifierDotAttribute
		The fully qualified module path containing the dataclass definition.
	dataclassIdentifier : str
		The name of the dataclass to decompose.
	instanceIdentifier : str
		The variable name to use for the dataclass instance in generated code.

	Returns
	-------
	ShatteredDataclass
		A ShatteredDataclass containing AST representations of all dataclass components,
		with imports, field definitions, annotations, and repackaging code.

	Raises
	------
	ValueError
		If the dataclass cannot be found in the specified module or if no counting variable is identified in the dataclass.

	"""
	Official_fieldOrder: list[str] = []
	dictionaryDeReConstruction: dict[str, DeReConstructField2ast] = {}

	dataclassClassDef: ast.ClassDef | None = extractClassDef(parseLogicalPath2astModule(logicalPathModule), dataclassIdentifier)
	if not dataclassClassDef:
		message = f"I could not find `{dataclassIdentifier = }` in `{logicalPathModule = }`."
		raise ValueError(message)

	countingVariable = None
	for aField in dataclasses.fields(importLogicalPath2Identifier(logicalPathModule, dataclassIdentifier)): # pyright: ignore [reportArgumentType]
		Official_fieldOrder.append(aField.name)
		dictionaryDeReConstruction[aField.name] = DeReConstructField2ast(logicalPathModule, dataclassClassDef, instanceIdentifier, aField)
		if aField.metadata.get('theCountingIdentifier', False):
			countingVariable = dictionaryDeReConstruction[aField.name].name
	if countingVariable is None:
		message = f"I could not find the counting variable in `{dataclassIdentifier = }` in `{logicalPathModule = }`."
		raise ValueError(message)

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
	shatteredDataclass.repack = Make.Assign([Make.Name(instanceIdentifier)], value=Make.Call(Make.Name(dataclassIdentifier), list_keyword=shatteredDataclass.list_keyword_field__field4init))
	shatteredDataclass.signatureReturnAnnotation = Make.Subscript(Make.Name('tuple'), Make.Tuple(shatteredDataclass.listAnnotations))

	shatteredDataclass.imports.update(*(dictionaryDeReConstruction[field].ledger for field in Official_fieldOrder))
	shatteredDataclass.imports.addImportFrom_asStr(logicalPathModule, dataclassIdentifier)

	return shatteredDataclass

def removeDataclassFromFunction(ingredientsTarget: IngredientsFunction, shatteredDataclass: ShatteredDataclass) -> IngredientsFunction:
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
	ingredientsTarget : IngredientsFunction
		The function definition and its dependencies to be transformed.
	shatteredDataclass : ShatteredDataclass
		The decomposed dataclass components providing AST mappings and transformations.

	Returns
	-------
	IngredientsFunction
		The modified function ingredients with dataclass dependencies removed.

	"""
	ingredientsTarget.astFunctionDef.args = Make.arguments(list_arg=shatteredDataclass.list_argAnnotated4ArgumentsSpecification)
	ingredientsTarget.astFunctionDef.returns = shatteredDataclass.signatureReturnAnnotation
	changeReturnCallable = NodeChanger(Be.Return, Then.replaceWith(Make.Return(shatteredDataclass.fragments4AssignmentOrParameters)))
	changeReturnCallable.visit(ingredientsTarget.astFunctionDef)
	ingredientsTarget.astFunctionDef = unparseFindReplace(ingredientsTarget.astFunctionDef, shatteredDataclass.map_stateDOTfield2Name)
	return ingredientsTarget

def unpackDataclassCallFunctionRepackDataclass(ingredientsCaller: IngredientsFunction, targetCallableIdentifier: str, shatteredDataclass: ShatteredDataclass) -> IngredientsFunction:
	"""Transform a caller function to interface with a dataclass-free target function.

	(AI generated docstring)

	This function complements `removeDataclassFromFunction` by modifying calling code to work with
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
	ingredientsCaller : IngredientsFunction
		The calling function definition and its dependencies to be transformed.
	targetCallableIdentifier : str
		The name of the target function being called.
	shatteredDataclass : ShatteredDataclass
		The decomposed dataclass components providing unpacking and repacking logic.

	Returns
	-------
	IngredientsFunction
		The modified caller function with appropriate unpacking and repacking around the target call.

	"""
	astCallTargetCallable = Make.Call(Make.Name(targetCallableIdentifier), shatteredDataclass.listName4Parameters)
	replaceAssignTargetCallable = NodeChanger(Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier)), Then.replaceWith(Make.Assign([shatteredDataclass.fragments4AssignmentOrParameters], value=astCallTargetCallable)))
	unpack4targetCallable = NodeChanger(Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier)), Then.insertThisAbove(shatteredDataclass.listUnpack))
	repack4targetCallable = NodeChanger(Be.Assign.valueIs(IfThis.isCallIdentifier(targetCallableIdentifier)), Then.insertThisBelow([shatteredDataclass.repack]))
	replaceAssignTargetCallable.visit(ingredientsCaller.astFunctionDef)
	unpack4targetCallable.visit(ingredientsCaller.astFunctionDef)
	repack4targetCallable.visit(ingredientsCaller.astFunctionDef)
	return ingredientsCaller
