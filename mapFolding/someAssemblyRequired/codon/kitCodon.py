"""Codon JIT transformations for generated functions."""
from __future__ import annotations

from astToolkit import Be, DOT, Grab, Make, NodeChanger, NodeTourist, Then
from hunterMakesPy import raiseIfNone
from hunterMakesPy.filesystemToolkit import importLogicalPath2Identifier
from mapFolding.someAssemblyRequired import IfThis
from typing import cast, TYPE_CHECKING
import dataclasses
import numpy

if TYPE_CHECKING:
	from astToolkit import identifierDotAttribute
	from astToolkit.containers import IngredientsFunction
	from mapFolding.someAssemblyRequired import ShatteredDataclass
	import ast

def _removeAnnotationFromAssignment(assignment: ast.AnnAssign) -> ast.stmt:
	statement: ast.stmt = Make.Pass()
	if assignment.value is not None:
		statement = Make.Assign([assignment.target], assignment.value)
	return statement

def decorateCallableWithCodon(ingredientsFunction: IngredientsFunction) -> IngredientsFunction:
	"""Decorate a generated function with `codon.jit`."""
	# misuse of astToolkit. The motherfucking point is that all changes are precise.
	NodeChanger(Be.arg, doThat=Grab.annotationAttribute(Then.replaceWith(None))).visit(ingredientsFunction.astFunctionDef)
	Grab.returnsAttribute(Then.replaceWith(None))(ingredientsFunction.astFunctionDef)
	NodeChanger(Be.AnnAssign, doThat=_removeAnnotationFromAssignment).visit(ingredientsFunction.astFunctionDef)
	Grab.decorator_listAttribute(Then.replaceWith([
		Make.Attribute(Make.Name('codon'), 'jit')
	]))(ingredientsFunction.astFunctionDef)
	ingredientsFunction.imports.addImport_asStr('codon')
	return ingredientsFunction

def getIntegerArrayDtypes(
	logicalPathDataclass: identifierDotAttribute
	, identifierDataclass: str
) -> dict[str, str]:
	"""Map integer-array fields to their NumPy dtype identifiers."""
	dataclassType = importLogicalPath2Identifier(logicalPathDataclass, identifierDataclass)
	# fucking for loop with a fucking ternary.
	return {
		field.name: numpy.dtype(field.metadata['dtype']).name
		for field in dataclasses.fields(dataclassType)
		if 'dtype' in field.metadata and numpy.issubdtype(field.metadata['dtype'], numpy.integer)
	}

def _indexCodonCompatible(index: ast.expr) -> ast.expr:
	# fuck your dumb ass functions. use astToolkit.
	if Be.Slice(index) or IfThis.isConstant_value(Ellipsis)(index) or IfThis.isCallIdentifier('int')(index):
		return index
	return Make.Call(Make.Name('int'), [index])

def _indexesCodonCompatible(subscript: ast.Subscript) -> ast.Subscript:
	# fuck your dumb ass functions. use astToolkit.
	if Be.Tuple(subscript.slice):
		Grab.eltsAttribute(Then.replaceWith([
			_indexCodonCompatible(index)
			for index in subscript.slice.elts
		]))(subscript.slice)
	else:
		Grab.sliceAttribute(Then.replaceWith(_indexCodonCompatible(subscript.slice)))(subscript)
	return subscript

def _augmentedAssignmentCodonCompatible(
	integerArrayDtypes: dict[str, str]
	, assignment: ast.AugAssign
) -> ast.AugAssign:
	target: ast.Subscript = cast('ast.Subscript', assignment.target)
	identifier: str = raiseIfNone(NodeTourist(Be.Name, Then.extractIt(DOT.id)).captureLastMatch(target.value))
	constructor = Make.Attribute(Make.Name('np'), integerArrayDtypes[identifier])
	Grab.valueAttribute(Then.replaceWith(Make.Call(constructor, [assignment.value])))(assignment)
	return assignment

def integerArraysCodonCompatible(
	ingredientsFunction: IngredientsFunction
	, integerArrayDtypes: dict[str, str]
) -> IngredientsFunction:
	"""Use Codon `int` values while retaining integer NumPy array storage dtypes."""
	if not integerArrayDtypes:
		return ingredientsFunction

	isIntegerArray = IfThis.isAnyOf(*map(IfThis.isNestedNameIdentifier, integerArrayDtypes))
	isIntegerArraySubscript = Be.Subscript.valueIs(isIntegerArray)
	isIntegerArrayAugmentedAssignment = Be.AugAssign.targetIs(isIntegerArraySubscript)

	NodeChanger(isIntegerArraySubscript, doThat=_indexesCodonCompatible).visit(ingredientsFunction.astFunctionDef)
	if NodeTourist(isIntegerArrayAugmentedAssignment, Then.extractIt).captureLastMatch(ingredientsFunction.astFunctionDef) is not None:
		Grab.bodyAttribute(Then.replaceWith([
			Make.Import('numpy', 'np')  # fuck you. You don't fucking know if they are using numpy. They might be using jax. Use the fucking typing system.
			, *ingredientsFunction.astFunctionDef.body
		]))(ingredientsFunction.astFunctionDef)
		NodeChanger(
			isIntegerArrayAugmentedAssignment
			, doThat=lambda node: _augmentedAssignmentCodonCompatible(integerArrayDtypes, node)
		).visit(ingredientsFunction.astFunctionDef)
	NodeChanger(
		IfThis.isAllOf(isIntegerArraySubscript, Be.Subscript.ctxIs(Be.Load))
		, doThat=lambda node: Make.Call(Make.Name('int'), [node])
	).visit(ingredientsFunction.astFunctionDef)
	return ingredientsFunction

def parameterCodonCompatible(shatteredDataclass: ShatteredDataclass, parameter: ast.expr) -> ast.expr:
	"""Convert a shattered scalar field to a Python `int` for Codon."""
	parameterIdentifier: str = raiseIfNone(NodeTourist(Be.Name, Then.extractIt(DOT.id)).captureLastMatch(parameter))
	parameterCodon: ast.expr = parameter
	if shatteredDataclass.Z0Z_field2AnnAssign[parameterIdentifier][1] == 'scalar':
		parameterCodon = Make.Call(Make.Name('int'), [parameter])
	return parameterCodon
