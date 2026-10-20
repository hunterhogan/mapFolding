"""Write specialized counting jobs and compile source on Linux.

(AI generated docstring)

You can use this module to embed state values and insert datatype conversions for generated
counting jobs. On Linux, generation also calls `mapFolding.kitAST.linux.toCodon` [1].

Contents
--------
Functions
	makeJob
		Write a specialized counting job and compile the source on Linux.
	variableCompatibility
		Insert datatype conversions around selected field operations.
Variables
	boxOfSettings形
		Access the fixed datatype substitutions used during source preparation.

References
----------
[1] `mapFolding.kitAST.linux.toCodon`
"""
from __future__ import annotations

from astToolkit import Be, Grab, Make, NodeChanger, Then
from astToolkit.containers import astModuleToIngredientsFunction, IngredientsFunction, IngredientsModule
from hunterMakesPy import raiseIfNone
from mapFolding.kitAST import IfThis, Settings形
from mapFolding.kitAST.RecipeJob import (
	addLauncher, fromMapShape, move_argToBody, RecipeJobTheorem2, replaceStaticScalars, setDatatypeViaImport)
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
	Settings形(datatypeIdentifier='形ArrayTotalLeaves1D', typeModule='numpy', typeIdentifier='uint8', type_asname='形ArrayTotalLeaves1D'),
	Settings形(datatypeIdentifier='形ArrayElephino1D', typeModule='numpy', typeIdentifier='uint8', type_asname='形ArrayElephino1D'),
	Settings形(datatypeIdentifier='形ArrayTotalLeaves3D', typeModule='numpy', typeIdentifier='uint8', type_asname='形ArrayTotalLeaves3D'),
]
"""Access the fixed datatype substitutions used during source preparation.

(AI generated docstring)

This list records imported scalar types and array element types while preserving abstract
aliases used by the source algorithm.
"""

def makeJob(job: RecipeJobTheorem2) -> PurePosixPath:
	"""Write a specialized counting job and compile the source on Linux.

	(AI generated docstring)

	You can use this function to specialize `job` by embedding state values, inserting datatype
	conversions, replacing datatype imports, and adding a standalone launcher.
	The function writes the Python source. On Linux, the function also calls
	`mapFolding.kitAST.linux.toCodon` [1], which rewrites that source and builds an executable.
	The function returns the configured source path on every platform.

	Parameters
	----------
	job : RecipeJobTheorem2
		Initialized recipe with source syntax, field metadata, and job output paths.

	Returns
	-------
	pathFilenameModule : PurePosixPath
		Configured Python source path; the Linux executable has the source suffix removed.

	Examples
	--------
	The module `mapFolding.kitAST.codon.makeJob` uses this call [2].

		```python
		makeJob(fromMapShape(mapShape))
		```

	References
	----------
	[1] `mapFolding.kitAST.linux.toCodon`

	[2] `mapFolding.kitAST.codon.makeJob`
	"""
	ingredientsCount: IngredientsFunction = astModuleToIngredientsFunction(raiseIfNone(job.source_astModule), job.identifierCallableSource)
	ingredientsCount.astFunctionDef.decorator_list = []

	replaceStaticScalars(job, ingredientsCount)

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
	"""Insert datatype conversions around selected field operations.

	(AI generated docstring)

	You can use this function to adapt `ingredientsFunction` for compiled execution by using
	field annotations from `job`. The function inserts conversions in selected assignments,
	arithmetic operands, comparisons, and indices. The original function container is modified
	in place and returned.

	Conversion Selection
	--------------------
	Matching augmented assignments and constant-valued assignments wrap the assigned value in
	the field annotation constructor. Binary operations with a matching left operand wrap the
	right operand. Matching comparisons retain one converted comparator. Matching subscripts
	convert one index, or each tuple index, with `int`.

	Parameters
	----------
	ingredientsFunction : IngredientsFunction
		Function container whose field operations will receive conversions.
	job : RecipeJobTheorem2
		Recipe containing annotated field arguments in decomposed dataclass metadata.

	Returns
	-------
	ingredientsFunction : IngredientsFunction
		The original container holding the transformed function syntax.

	Examples
	--------
	The module `mapFolding.kitAST.codon.makeJob` uses this call [1].

		```python
		variableCompatibility(ingredientsCount, job)
		```

	References
	----------
	[1] `mapFolding.kitAST.codon.makeJob`
	"""
	for ast_arg in raiseIfNone(job.shatteredDataclass).boxOf_argAnnotated:
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
					Make.Call(Make.Name('int'), listParameters=[cast('ast.Tuple', node.slice).elts[index]])
					for index in range(len(cast('ast.Tuple', node.slice).elts))])))(node)
		).visit(ingredientsFunction.astFunctionDef)

	return ingredientsFunction

if __name__ == '__main__':
	mapShape: tuple[形TotalLeaves, ...] = (2, 14)
	makeJob(fromMapShape(mapShape))
