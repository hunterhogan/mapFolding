"""Codon JIT transformations for generated functions."""
from __future__ import annotations

from astToolkit import Be, DOT, Grab, Make, NodeChanger, Then
from astToolkit.containers import IngredientsFunction  # ruff:ignore[typing-only-third-party-import]
from collections.abc import Iterator, Sequence  # ruff:ignore[typing-only-standard-library-import]
from copy import deepcopy
from mapFolding.someAssemblyRequired import IfThis
from more_itertools import unique_everseen
import ast  # ruff:ignore[typing-only-standard-library-import]

identifierCompatibleValue: str = 'compatibleValue'

def _referenceAsLoad(reference: ast.expr) -> ast.expr:
	referenceLoad: ast.expr = deepcopy(reference)
	NodeChanger(Be.Name, Grab.ctxAttribute(Then.replaceWith(Make.Load()))).visit(referenceLoad)
	NodeChanger(Be.Attribute, Grab.ctxAttribute(Then.replaceWith(Make.Load()))).visit(referenceLoad)
	NodeChanger(Be.Subscript, Grab.ctxAttribute(Then.replaceWith(Make.Load()))).visit(referenceLoad)
	NodeChanger(Be.Starred, Grab.ctxAttribute(Then.replaceWith(Make.Load()))).visit(referenceLoad)
	return referenceLoad

def _valueCompatible(reference: ast.expr, value: ast.expr) -> ast.Call:
	return Make.Call(Make.Name(identifierCompatibleValue), [_referenceAsLoad(reference), value])

def _assignmentValueCompatible(assignment: ast.Assign) -> ast.Assign:
	reference: ast.expr = assignment.targets[0]
	Grab.valueAttribute(Then.replaceWith(
		_valueCompatible(reference, assignment.value)
	))(assignment)
	return assignment

def _augmentedAssignmentValueCompatible(assignment: ast.AugAssign) -> ast.AugAssign:
	Grab.valueAttribute(Then.replaceWith(
		_valueCompatible(assignment.target, assignment.value)
	))(assignment)
	return assignment

def _binaryOperationValueCompatible(operation: ast.BinOp) -> ast.BinOp:
	return Grab.rightAttribute(Then.replaceWith(
		_valueCompatible(operation.left, operation.right)
	))(operation)

def _comparisonValueCompatible(comparison: ast.Compare) -> ast.Compare:
	references: list[ast.expr] = [comparison.left, *comparison.comparators[:-1]]
	return Grab.comparatorsAttribute(Then.replaceWith(list(map(
		_valueCompatible, references, comparison.comparators
	))))(comparison)

def _indexCompatible(index: ast.expr) -> ast.expr:
	isAlreadyCompatible = IfThis.isAnyOf(
		Be.Slice
		, Be.Constant
		, Be.Call.funcIs(Be.Attribute.attrIs(IfThis.isIdentifier('__index__')))
	)
	if isAlreadyCompatible(index):
		return index
	if Be.Subscript(index):
		_subscriptIndexesCompatible(index)
	return Make.Call(Make.Attribute(index, '__index__'))

def _indexesCompatible(indexes: list[ast.expr]) -> list[ast.expr]:
	return list(map(_indexCompatible, indexes))

def _singleIndexCompatible(subscript: ast.Subscript) -> ast.Subscript:
	return Grab.sliceAttribute(_indexCompatible)(subscript)

def _tupleIndexesCompatible(subscript: ast.Subscript) -> ast.Subscript:
	return Grab.sliceAttribute(Grab.eltsAttribute(_indexesCompatible))(subscript)

def _subscriptIndexesCompatible(subscript: ast.Subscript) -> ast.Subscript:
	if Be.Subscript(subscript.value):
		_subscriptIndexesCompatible(subscript.value)
	if Be.Tuple(subscript.slice):
		return _tupleIndexesCompatible(subscript)
	return _singleIndexCompatible(subscript)

def _makeCompatibleValueFunction() -> ast.FunctionDef:
	identifierReference: str = 'Reference'
	identifierValue: str = 'Value'
	return Make.FunctionDef(
		identifierCompatibleValue
		, Make.arguments(list_arg=[
			Make.arg('_reference', Make.Name(identifierReference))
			, Make.arg('value', Make.Name(identifierValue))
		])
		, [Make.Return(Make.Call(Make.Name(identifierReference), [Make.Name('value')]))]
		, returns=Make.Name(identifierReference)
		, type_params=[
			Make.TypeVar(identifierReference)
			, Make.TypeVar(identifierValue)
		]
	)

def _nameAnnotation(parameter: ast.arg) -> ast.Name | None:
	annotation: ast.expr | None = parameter.annotation
	if annotation is not None and Be.Name(annotation):
		return annotation
	return None

def _parameterTypeIdentifiers(parameters: Sequence[ast.arg]) -> Iterator[str]:
	annotations: Iterator[ast.Name] = filter(None, map(_nameAnnotation, parameters))
	return unique_everseen(map(DOT.id, annotations))

def decorateCallableWithCodon(ingredientsFunction: IngredientsFunction) -> IngredientsFunction:
	"""Decorate a generated function with `codon.jit` and its abstract type parameters."""
	Grab.type_paramsAttribute(Then.replaceWith(list(map(
		Make.TypeVar, _parameterTypeIdentifiers(ingredientsFunction.astFunctionDef.args.args)
	))))(ingredientsFunction.astFunctionDef)
	Grab.decorator_listAttribute(Then.replaceWith([
		Make.Attribute(Make.Name('codon'), 'jit')
	]))(ingredientsFunction.astFunctionDef)
	ingredientsFunction.imports.addImportFrom_asStr('__future__', 'annotations')
	ingredientsFunction.imports.addImport_asStr('codon')
	return ingredientsFunction

def variableCompatibility(
	ingredientsFunction: IngredientsFunction
	, parameters: Sequence[ast.arg]
) -> IngredientsFunction:
	"""Preserve each established variable type through operations and indexing."""
	isParameter = IfThis.isAnyOf(*map(
		IfThis.isNestedNameIdentifier, map(DOT.arg, parameters)
	))
	isParameterSubscript = Be.Subscript.valueIs(isParameter)
	functionBody = Make.Module(ingredientsFunction.astFunctionDef.body)

	NodeChanger(
		IfThis.isAssignAndTargets0Is(isParameter)
		, _assignmentValueCompatible
	).visit(functionBody)
	NodeChanger(
		Be.AugAssign.targetIs(isParameter)
		, _augmentedAssignmentValueCompatible
	).visit(functionBody)
	NodeChanger(
		Be.BinOp.leftIs(isParameter)
		, _binaryOperationValueCompatible
	).visit(functionBody)
	NodeChanger(
		Be.Compare.leftIs(isParameter)
		, _comparisonValueCompatible
	).visit(functionBody)
	NodeChanger(isParameterSubscript, _subscriptIndexesCompatible).visit(functionBody)

	Grab.bodyAttribute(Then.replaceWith([
		_makeCompatibleValueFunction()
		, *ingredientsFunction.astFunctionDef.body
	]))(ingredientsFunction.astFunctionDef)
	return ingredientsFunction
