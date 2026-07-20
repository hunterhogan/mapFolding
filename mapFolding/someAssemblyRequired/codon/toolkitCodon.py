"""Codon just-in-time compilation transformations."""

from __future__ import annotations

from astToolkit import Be, Grab, Make, NodeChanger, Then
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from astToolkit.containers import IngredientsFunction
	from mapFolding.someAssemblyRequired import ShatteredDataclass
	import ast

identifierModuleCodon: str = 'codon'
identifierDecoratorCodon: str = 'jit'
identifierModuleNumpy: str = 'numpy'

def removeAnnotationFromAssignment(assignment: ast.AnnAssign) -> ast.stmt:
	"""Convert a local annotated assignment into Codon-inferable syntax."""
	if assignment.value is None:
		return Make.Pass()
	return Make.Assign([assignment.target], assignment.value)

def castParameterScalarForCodon(shatteredDataclass: ShatteredDataclass, parameter: ast.Name) -> ast.expr:
	"""Convert NumPy scalar fields to native Python scalars at the Codon boundary."""
	parameterCodon: ast.expr = parameter
	if shatteredDataclass.Z0Z_field2AnnAssign[parameter.id][1] == 'scalar':
		parameterCodon = Make.IfExp(
			Make.Call(
				Make.Name('isinstance')
				, listParameters=[Make.Name(parameter.id), Make.Attribute(Make.Name(identifierModuleNumpy), 'generic')]
			)
			, Make.Call(Make.Attribute(Make.Name(parameter.id), 'item'))
			, Make.Name(parameter.id)
		)
	return parameterCodon

def decorateCallableWithCodon(ingredientsFunction: IngredientsFunction) -> IngredientsFunction:
	"""Make a function independently compilable by ``codon.jit``."""
	NodeChanger(
		Be.arg
		, doThat=Grab.annotationAttribute(Then.replaceWith(None))
	).visit(ingredientsFunction.astFunctionDef)
	Grab.returnsAttribute(Then.replaceWith(None))(ingredientsFunction.astFunctionDef)
	NodeChanger(Be.AnnAssign, doThat=removeAnnotationFromAssignment).visit(ingredientsFunction.astFunctionDef)
	Grab.decorator_listAttribute(Then.replaceWith([
		Make.Attribute(Make.Name(identifierModuleCodon), identifierDecoratorCodon)
	]))(ingredientsFunction.astFunctionDef)
	ingredientsFunction.imports.addImport_asStr(identifierModuleCodon)
	return ingredientsFunction
