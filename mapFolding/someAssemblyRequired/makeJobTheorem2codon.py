"""codon.

https://docs.exaloop.io/start/install/
"""
from astToolkit import (
	Be, extractFunctionDef, Grab, identifierDotAttribute, Make, NodeChanger, parseLogicalPath2astModule, Then)
from astToolkit.containers import IngredientsFunction, IngredientsModule
from astToolkit.transformationTools import removeUnusedParameters, write_astModule
from hunterMakesPy import raiseIfNone
from mapFolding import DatatypeLeavesTotal, getPathFilenameFoldsTotal, packageSettings
from mapFolding.dataBaskets import MapFoldingState
from mapFolding.someAssemblyRequired import DatatypeConfiguration, default, IfThis
from mapFolding.someAssemblyRequired.RecipeJob import (
	customizeDatatypeViaImport, moveShatteredDataclass_arg2body, RecipeJobTheorem2)
from mapFolding.syntheticModules.initializeState import transitionOnGroupsOfFolds
from pathlib import Path, PurePosixPath
from typing import cast, TYPE_CHECKING
import subprocess
import sys

if TYPE_CHECKING:
	from io import TextIOBase
	import ast

# TODO Converge with `makeJobTheorem2Numba`.

listDatatypeConfigurations: list[DatatypeConfiguration] = [
	DatatypeConfiguration(datatypeIdentifier='DatatypeLeavesTotal', typeModule='numpy', typeIdentifier='uint8', type_asname='DatatypeLeavesTotal'),
	DatatypeConfiguration(datatypeIdentifier='DatatypeElephino', typeModule='numpy', typeIdentifier='uint8', type_asname='DatatypeElephino'),
	DatatypeConfiguration(datatypeIdentifier='DatatypeFoldsTotal', typeModule='numpy', typeIdentifier='int64', type_asname='DatatypeFoldsTotal'),
	DatatypeConfiguration(datatypeIdentifier='Array1DLeavesTotal', typeModule='numpy', typeIdentifier='uint8', type_asname='Array1DLeavesTotal'),
	DatatypeConfiguration(datatypeIdentifier='Array1DElephino', typeModule='numpy', typeIdentifier='uint8', type_asname='Array1DElephino'),
	DatatypeConfiguration(datatypeIdentifier='Array3DLeavesTotal', typeModule='numpy', typeIdentifier='uint8', type_asname='Array3DLeavesTotal'),
]

def _addWriteFoldsTotal(ingredientsFunction: IngredientsFunction, job: RecipeJobTheorem2) -> IngredientsFunction:
	NodeChanger(Be.Return, Then.removeIt).visit(ingredientsFunction.astFunctionDef)
	ingredientsFunction.astFunctionDef.returns = Make.Constant(None)

	writeFoldsTotal = Make.Expr(Make.Call(Make.Attribute(
		Make.Call(Make.Name('open'), listParameters=[Make.Constant(str(job.pathFilenameFoldsTotal.as_posix())), Make.Constant('w')])
		, 'write'), listParameters=[Make.Call(Make.Name('str'), listParameters=[
			Make.Mult().join([job.shatteredDataclass.countingVariableName, Make.Constant(job.state.leavesTotal * 2)])])]))

	NodeChanger(IfThis.isAllOf(Be.AugAssign.targetIs(IfThis.isNameIdentifier(job.shatteredDataclass.countingVariableName.id))
			, Be.AugAssign.opIs(Be.Mult), Be.AugAssign.valueIs(Be.Constant))
		, doThat=Then.replaceWith(writeFoldsTotal)
	).visit(ingredientsFunction.astFunctionDef)

	return ingredientsFunction

def _variableCompatibility(ingredientsFunction: IngredientsFunction, job: RecipeJobTheorem2) -> IngredientsFunction:
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
	for ast_arg in job.shatteredDataclass.list_argAnnotated4ArgumentsSpecification:
		identifier: str = ast_arg.arg
		annotation: ast.expr = raiseIfNone(ast_arg.annotation)

	# ------- `identifier` is target of Augmented Assignment, or --------------
	# ------- `identifier` is target of Assignment and value is Constant. -----
		NodeChanger(
			IfThis.isAnyOf(
							Be.AugAssign.targetIs(IfThis.isNestedNameIdentifier(identifier))
			,   IfThis.isAllOf(Be.Assign.targetsIs(Be.at(0, IfThis.isNestedNameIdentifier(identifier)))
							,  Be.Assign.valueIs(Be.Constant))
			)
			, doThat=lambda node, annotation=annotation: Grab.valueAttribute(Then.replaceWith(Make.Call(annotation, listParameters=[node.value])))(node)
		).visit(ingredientsFunction.astFunctionDef)

	# ------- `identifier` - 1. ----------------------------------------------
		NodeChanger(Be.BinOp.leftIs(IfThis.isNestedNameIdentifier(identifier))
			, doThat=lambda node, annotation=annotation: Grab.rightAttribute(Then.replaceWith(Make.Call(annotation, listParameters=[node.right])))(node)
		).visit(ingredientsFunction.astFunctionDef)

	# ------- `identifier` in Comparison. -------------------------------------
		NodeChanger(Be.Compare.leftIs(IfThis.isNestedNameIdentifier(identifier))
			, doThat=lambda node, annotation=annotation: Grab.comparatorsAttribute(lambda at, annotation=annotation: Then.replaceWith([Make.Call(annotation, listParameters=[node.comparators[0]])])(at[0]))(node)
		).visit(ingredientsFunction.astFunctionDef)

	# ------- `identifier` has exactly one index value. -----------------------
		NodeChanger(IfThis.isAllOf(Be.Subscript.valueIs(IfThis.isNestedNameIdentifier(identifier))
			, lambda node: not Be.Subscript.sliceIs(Be.Tuple)(node))
			, doThat=lambda node: Grab.sliceAttribute(Then.replaceWith(Make.Call(Make.Name('int'), listParameters=[node.slice])))(node)
		).visit(ingredientsFunction.astFunctionDef)

	# ------- `identifier` has multiple index values. -------------------------
		NodeChanger(IfThis.isAllOf(Be.Subscript.valueIs(IfThis.isNestedNameIdentifier(identifier))
								,  Be.Subscript.sliceIs(Be.Tuple))
			, doThat=lambda node: Grab.sliceAttribute(Grab.eltsAttribute(
				Then.replaceWith([
					Make.Call(Make.Name('int'), listParameters=[cast(ast.Tuple, node.slice).elts[index]])
					for index in range(len(cast(ast.Tuple, node.slice).elts))])))(node)
		).visit(ingredientsFunction.astFunctionDef)

	return ingredientsFunction

def makeJob(job: RecipeJobTheorem2) -> None:
	"""Generate an optimized module for map folding calculations.

	This function orchestrates the complete code transformation assembly line to convert a generic map folding algorithm into a
	highly optimized, specialized computation module.

	Parameters
	----------
	job : RecipeJobTheorem2
		Configuration recipe containing source locations, target paths, raw materials, and state.

	"""
	ingredientsCount: IngredientsFunction = IngredientsFunction(raiseIfNone(extractFunctionDef(job.source_astModule, job.identifierCallable)))
	ingredientsCount.astFunctionDef.decorator_list = []

	# Replace identifiers-with-static-values with their values.
	for identifier in job.shatteredDataclass.listIdentifiersStaticScalars:
		NodeChanger(IfThis.isNameIdentifier(identifier)
			, Then.replaceWith(Make.Constant(int(eval(f"job.state.{identifier}"))))  # noqa: S307
		).visit(ingredientsCount.astFunctionDef)

	ingredientsCount.imports.update(job.shatteredDataclass.imports)
	ingredientsCount.removeUnusedParameters()
	NodeChanger(Be.arg, lambda removeIt: ingredientsCount.astFunctionDef.body.insert(0, moveShatteredDataclass_arg2body(removeIt.arg, job))).visit(ingredientsCount.astFunctionDef)

	ingredientsCount = _addWriteFoldsTotal(ingredientsCount, job)
	ingredientsCount = _variableCompatibility(ingredientsCount, job)

	ingredientsModule = IngredientsModule(launcher=Make.Module([
		Make.If(Make.Compare(Make.Name('__name__'), [Make.Eq()], [Make.Constant('__main__')])
			, body=[Make.Expr(Make.Call(Make.Name(job.identifierCallable)))])]))

	ingredientsCount, ingredientsModule = customizeDatatypeViaImport(ingredientsCount, ingredientsModule, listDatatypeConfigurations)

	ingredientsCount.imports.removeImportFromModule('mapFolding.dataBaskets')

	ingredientsModule.appendIngredientsFunction(ingredientsCount)

	if sys.platform == 'linux':
		Path(job.pathFilenameModule).parent.mkdir(parents=True, exist_ok=True)
		buildCommand: list[str] = ['codon', 'build', '--exe', '--release',
			'--fast-math', '--enable-unsafe-fp-math', '--disable-exceptions',
			'--mcpu=native',
			'-o', str(job.pathFilenameModule.with_suffix('')),
			'-']
		streamText = subprocess.Popen(buildCommand, stdin=subprocess.PIPE, text=True)
		if streamText.stdin is not None:
			write_astModule(ingredientsModule, pathFilename=cast(TextIOBase, streamText.stdin), packageName=job.packageIdentifier)
			streamText.stdin.close()
		streamText.wait()
		subprocess.run(['/usr/bin/strip', str(job.pathFilenameModule.with_suffix(''))], check=False)
		sys.stdout.write(f"sudo systemd-run --unit={job.moduleIdentifier} --nice=-10 --property=CPUAffinity=0 {job.pathFilenameModule.with_suffix('')}\n")
	else:
		ingredientsModule.write_astModule(job.pathFilenameModule, identifierPackage=job.packageIdentifier or '')
		sys.stdout.write(f"python {Path(job.pathFilenameModule)}\n")

def fromMapShape(mapShape: tuple[DatatypeLeavesTotal, ...]) -> None:
	"""Create a binary executable for a map-folding job from map dimensions.

	This function initializes a map folding computation state from the given map shape, sets up the necessary file paths, and
	generates an optimized executable for the specific map configuration.

	Parameters
	----------
	mapShape : tuple[DatatypeLeavesTotal, ...]
		Dimensions of the map as a tuple where each element represents the size
		along one axis.

	"""
	state: MapFoldingState = transitionOnGroupsOfFolds(MapFoldingState(mapShape))
	pathModule = PurePosixPath(Path.home(), 'mapFolding', 'jobs')
	logicalPath2astModule: identifierDotAttribute = f'{packageSettings.identifierPackage}.{default['logicalPath']['synthetic']}.theorem2Numba'
	source_astModule: ast.Module = parseLogicalPath2astModule(logicalPath2astModule)
	pathFilenameFoldsTotal = PurePosixPath(getPathFilenameFoldsTotal(state.mapShape, pathModule))
	aJob = RecipeJobTheorem2(state, source_astModule=source_astModule, pathModule=pathModule, pathFilenameFoldsTotal=pathFilenameFoldsTotal)
	makeJob(aJob)

if __name__ == '__main__':
	mapShape: tuple[DatatypeLeavesTotal, ...] = (1, 14)
	fromMapShape(mapShape)

