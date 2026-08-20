"""mapFolding job."""
from __future__ import annotations

from astToolkit import Be, Grab, Make, NodeChanger, Then
from astToolkit.containers import astModuleToIngredientsFunction, IngredientsFunction, IngredientsModule
from hunterMakesPy import raiseIfNone
from mapFolding.kitAST import IfThis, Settings形
from mapFolding.kitAST.RecipeJob import addLauncher, fromMapShape, move_argToBody, RecipeJobTheorem2, setDatatypeViaImport, staticValues
from mapFolding.theTypes import 形TotalLeaves  # ruff: ignore[typing-only-first-party-import]
from pathlib import Path, PurePosixPath
from typing import cast, TYPE_CHECKING
import sys

if TYPE_CHECKING:
	import ast

boxOfSettings形: list[Settings形] = [
	Settings形(datatypeIdentifier='形TotalLeaves', typeModule='numpy', typeIdentifier='uint8', type_asname='形TotalLeaves'),
	Settings形(datatypeIdentifier='形Elephino', typeModule='numpy', typeIdentifier='uint8', type_asname='形Elephino'),
	Settings形(datatypeIdentifier='形TotalFolds', typeModule='numpy', typeIdentifier='uint64', type_asname='形TotalFolds'),
	Settings形(datatypeIdentifier='形Array1DTotalLeaves', typeModule='numpy', typeIdentifier='uint8', type_asname='形Array1DTotalLeaves'),
	Settings形(datatypeIdentifier='形Array1DElephino', typeModule='numpy', typeIdentifier='uint8', type_asname='形Array1DElephino'),
	Settings形(datatypeIdentifier='形Array3DTotalLeaves', typeModule='numpy', typeIdentifier='uint8', type_asname='形Array3DTotalLeaves'),
]

def makeJob(job: RecipeJobTheorem2) -> PurePosixPath:
	"""Generate an optimized module for map folding calculations.

	This function orchestrates the complete code transformation assembly line to convert a generic map folding algorithm into a
	highly optimized, specialized computation module.

	Parameters
	----------
	job : RecipeJobTheorem2
		Configuration recipe containing source locations, target paths, raw materials, and state.

	"""
	ingredientsCount: IngredientsFunction = astModuleToIngredientsFunction(raiseIfNone(job.source_astModule), job.identifierCallableSource)
	ingredientsCount.astFunctionDef.decorator_list = []

	staticValues(job, ingredientsCount)

	ingredientsModule = IngredientsModule()
	addLauncher(ingredientsModule, ingredientsCount, job)
	ingredientsCount = variableCompatibility(ingredientsCount, job)
	ingredientsCount = move_argToBody(ingredientsCount, job)

	ingredientsCount, ingredientsModule = setDatatypeViaImport(ingredientsCount, ingredientsModule, boxOfSettings形)

	ingredientsModule.appendIngredientsFunction(ingredientsCount)

	ingredientsModule.write_astModule(job.pathFilenameModule, identifierPackage=job.identifierPackage or '')

	if sys.platform == 'linux':
		from mapFolding.kitAST.linux import toCodon  # ruff: ignore[import-outside-top-level]
		toCodon(Path(job.pathFilenameModule))

	return job.pathFilenameModule

def variableCompatibility(ingredientsFunction: IngredientsFunction, job: RecipeJobTheorem2) -> IngredientsFunction:
	"""Ensure the variable is compiled to the correct type.

	Add a type constructor to `identifier` to ensure compatibility if
	- an incompatible type might be assigned to it,
	- it might be compared with an incompatible type,
	- it is used as an indexer but its type is not a valid indexer type.

	Parameters
	----------
	ingredientsFunction : IngredientsFunction
		Function to modify.
	job : RecipeJobTheorem2
		Configuration settings with identifiers and their type annotations.

	Returns
	-------
	ingredientsFunction : IngredientsFunction
		Modified function.
	"""
	for ast_arg in raiseIfNone(job.shatteredDataclass).boxOf_argAnnotated4ArgumentsSpecification:
		identifier: str = ast_arg.arg
		annotation: ast.expr = raiseIfNone(ast_arg.annotation)

	#-------- `identifier` is target of Augmented Assignment, or --------------
	#-------- `identifier` is target of Assignment and value is Constant. -----
		NodeChanger(
			IfThis.isAnyOf(
							Be.AugAssign.targetIs(IfThis.isNestedNameIdentifier(identifier))
			, IfThis.isAllOf(Be.Assign.targetsIs(Be.at(0, IfThis.isNestedNameIdentifier(identifier)))
							, Be.Assign.valueIs(Be.Constant))
			)
			, doThat=lambda node, annotation=annotation: Grab.valueAttribute(Then.replaceWith(Make.Call(annotation, listParameters=[node.value])))(node)  # ty:ignore[unresolved-attribute, invalid-argument-type] # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType,reportArgumentType,reportAttributeAccessIssue]
		).visit(ingredientsFunction.astFunctionDef)

	#-------- `identifier` - 1. ----------------------------------------------
		NodeChanger(Be.BinOp.leftIs(IfThis.isNestedNameIdentifier(identifier))
			, doThat=lambda node, annotation=annotation: Grab.rightAttribute(Then.replaceWith(Make.Call(annotation, listParameters=[node.right])))(node)
		).visit(ingredientsFunction.astFunctionDef)

	#-------- `identifier` in Comparison. -------------------------------------
		NodeChanger(Be.Compare.leftIs(IfThis.isNestedNameIdentifier(identifier))
			, doThat=lambda node, annotation=annotation: Grab.comparatorsAttribute(lambda at, annotation=annotation: Then.replaceWith([Make.Call(annotation, listParameters=[node.comparators[0]])])(at[0]))(node)
		).visit(ingredientsFunction.astFunctionDef)

	#-------- `identifier` has exactly one index value. -----------------------
		NodeChanger(IfThis.isAllOf(Be.Subscript.valueIs(IfThis.isNestedNameIdentifier(identifier))
			, lambda node: not Be.Subscript.sliceIs(Be.Tuple)(node))
			, doThat=lambda node: Grab.sliceAttribute(Then.replaceWith(Make.Call(Make.Name('int'), listParameters=[node.slice])))(node)  # ty:ignore[unresolved-attribute, invalid-argument-type] # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType,reportArgumentType,reportAttributeAccessIssue]
		).visit(ingredientsFunction.astFunctionDef)

	#-------- `identifier` has multiple index values. -------------------------
		NodeChanger(IfThis.isAllOf(Be.Subscript.valueIs(IfThis.isNestedNameIdentifier(identifier))
								, Be.Subscript.sliceIs(Be.Tuple))
			, doThat=lambda node: Grab.sliceAttribute(Grab.eltsAttribute(
				Then.replaceWith([
					Make.Call(Make.Name('int'), listParameters=[cast('ast.Tuple', node.slice).elts[index]])  # ty:ignore[unresolved-attribute] # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType,reportArgumentType,reportAttributeAccessIssue]
					for index in range(len(cast('ast.Tuple', node.slice).elts))])))(node)  # ty:ignore[unresolved-attribute, invalid-argument-type] # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType,reportArgumentType,reportAttributeAccessIssue]
		).visit(ingredientsFunction.astFunctionDef)

	return ingredientsFunction

if __name__ == '__main__':
	mapShape: tuple[形TotalLeaves, ...] = (2, 14)
	makeJob(fromMapShape(mapShape))
