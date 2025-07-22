"""codon.

https://docs.exaloop.io/start/install/
"""

from astToolkit import (
	Be, DOT, extractFunctionDef, Grab, identifierDotAttribute, IngredientsFunction, IngredientsModule, Make, NodeChanger,
	NodeTourist, Then)
from astToolkit.transformationTools import removeUnusedParameters, write_astModule
from hunterMakesPy import autoDecodingRLE, raiseIfNone
from mapFolding import DatatypeLeavesTotal, getPathFilenameFoldsTotal, MapFoldingState
from mapFolding.someAssemblyRequired import IfThis
from mapFolding.someAssemblyRequired.RecipeJob import RecipeJobTheorem2
from mapFolding.syntheticModules.initializeCount import initializeGroupsOfFolds
from pathlib import Path, PurePosixPath
from typing import cast, NamedTuple, TYPE_CHECKING
import ast
import subprocess
import sys

if TYPE_CHECKING:
	from io import TextIOBase

class DatatypeConfiguration(NamedTuple):
	"""Configuration for mapping framework datatypes to compiled datatypes.

	This configuration class defines how abstract datatypes used in the map folding framework should be replaced with compiled
	datatypes during code generation. Each configuration specifies the source module, target type name, and optional import alias
	for the transformation.

	Attributes
	----------
	datatypeIdentifier : str
		Framework datatype identifier to be replaced.
	typeModule : identifierDotAttribute
		Module containing the target datatype (e.g., 'codon', 'numpy').
	typeIdentifier : str
		Concrete type name in the target module.
	type_asname : str | None = None
		Optional import alias for the type.
	"""

	datatypeIdentifier: str
	typeModule: identifierDotAttribute
	typeIdentifier: str
	type_asname: str | None = None

# TODO replace with dynamic system. Probably use `Final` in the dataclass.
listIdentifiersStaticValuesHARDCODED: list[str] = ['dimensionsTotal', 'leavesTotal']

# TODO Dynamically calculate the bitwidth of each datatype.
listDatatypeConfigurations: list[DatatypeConfiguration] = [
	DatatypeConfiguration(datatypeIdentifier='DatatypeLeavesTotal', typeModule='numpy', typeIdentifier='uint8', type_asname='DatatypeLeavesTotal'),
	DatatypeConfiguration(datatypeIdentifier='DatatypeElephino', typeModule='numpy', typeIdentifier='uint8', type_asname='DatatypeElephino'),
	DatatypeConfiguration(datatypeIdentifier='DatatypeFoldsTotal', typeModule='numpy', typeIdentifier='int64', type_asname='DatatypeFoldsTotal'),
]

listNumPy_dtype: list[DatatypeConfiguration] = [
	DatatypeConfiguration(datatypeIdentifier='Array1DLeavesTotal', typeModule='numpy', typeIdentifier='uint8', type_asname='Array1DLeavesTotal'),
	DatatypeConfiguration(datatypeIdentifier='Array1DElephino', typeModule='numpy', typeIdentifier='uint8', type_asname='Array1DElephino'),
	DatatypeConfiguration(datatypeIdentifier='Array3D', typeModule='numpy', typeIdentifier='uint8', type_asname='Array3D'),
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

def _datatypeDefinitions(ingredientsFunction: IngredientsFunction, ingredientsModule: IngredientsModule) -> tuple[IngredientsFunction, IngredientsModule]:
	for datatypeConfig in [*listDatatypeConfigurations, *listNumPy_dtype]:
		ingredientsFunction.imports.removeImportFrom(datatypeConfig.typeModule, None, datatypeConfig.datatypeIdentifier)
		ingredientsFunction.imports.addImportFrom_asStr(datatypeConfig.typeModule, datatypeConfig.typeIdentifier, datatypeConfig.type_asname)

	ingredientsFunction.imports.removeImportFromModule('mapFolding.dataBaskets')

	return ingredientsFunction, ingredientsModule

def _pythonCode2expr(string: str) -> ast.expr:
	"""Convert *one* expression as a string of Python code to an `ast.expr`."""
	return raiseIfNone(NodeTourist(Be.Expr, Then.extractIt(DOT.value)).captureLastMatch(ast.parse(string)))

def _variableCompatibility(ingredientsFunction: IngredientsFunction, job: RecipeJobTheorem2) -> IngredientsFunction:
	# On some assignment or comparison values, add a type constructor to ensure compatibility.
	# On some values-as-indexer, add a type constructor to ensure indexing-method compatibility.
	for ast_arg in job.shatteredDataclass.list_argAnnotated4ArgumentsSpecification:
		identifier = ast_arg.arg
		annotation = raiseIfNone(ast_arg.annotation)

		# `identifier` in Augmented Assignment, or in Assignments and value is Constant.
		NodeChanger(findThis=IfThis.isAnyOf(
				Be.AugAssign.targetIs(IfThis.isNestedNameIdentifier(identifier))
				, IfThis.isAllOf(IfThis.isAssignAndTargets0Is(IfThis.isNameIdentifier(identifier))
					, Be.Assign.valueIs(Be.Constant))
			)
			, doThat=lambda node, annotation=annotation: Grab.valueAttribute(Then.replaceWith(Make.Call(annotation, listParameters=[node.value])))(node)
		).visit(ingredientsFunction.astFunctionDef)

		# `identifier` - 1.
		NodeChanger(Be.BinOp.leftIs(IfThis.isNestedNameIdentifier(identifier))
			, doThat=lambda node, annotation=annotation: Grab.rightAttribute(Then.replaceWith(Make.Call(annotation, listParameters=[node.right])))(node)
		).visit(ingredientsFunction.astFunctionDef)

		# `identifier` in Comparison.
		NodeChanger(Be.Compare.leftIs(IfThis.isNestedNameIdentifier(identifier))
			, doThat=lambda node, annotation=annotation: Grab.comparatorsAttribute(lambda at, annotation=annotation: Then.replaceWith([Make.Call(annotation, listParameters=[node.comparators[0]])])(at[0]))(node)
		).visit(ingredientsFunction.astFunctionDef)

		# `identifier` has exactly one index value.
		NodeChanger(IfThis.isAllOf(Be.Subscript.valueIs(IfThis.isNestedNameIdentifier(identifier))
			, lambda node: not Be.Subscript.sliceIs(Be.Tuple)(node))
			, doThat=lambda node: Grab.sliceAttribute(Then.replaceWith(Make.Call(Make.Name('int'), listParameters=[node.slice])))(node)
		).visit(ingredientsFunction.astFunctionDef)

		# `identifier` has multiple index values.
		NodeChanger(IfThis.isAllOf(Be.Subscript.valueIs(IfThis.isNestedNameIdentifier(identifier))
			, Be.Subscript.sliceIs(Be.Tuple))
			, doThat=lambda node: Grab.sliceAttribute(Grab.eltsAttribute(
				Then.replaceWith([
					Make.Call(Make.Name('int'), listParameters=[cast('ast.Tuple', node.slice).elts[index]])
					for index in range(len(cast('ast.Tuple', node.slice).elts))])))(node)
		).visit(ingredientsFunction.astFunctionDef)

	return ingredientsFunction

def _move_arg2body(identifier: str, job: RecipeJobTheorem2) -> ast.AnnAssign | ast.Assign:
	Ima___Assign, elementConstructor = job.shatteredDataclass.Z0Z_field2AnnAssign[identifier]
	match elementConstructor:
		case 'scalar':
			cast('ast.Constant', cast('ast.Call', Ima___Assign.value).args[0]).value = int(job.state.__dict__[identifier])
		case 'array':
			dataAsStrRLE: str = autoDecodingRLE(job.state.__dict__[identifier], assumeAddSpaces=True)
			dataAs_ast_expr: ast.expr = _pythonCode2expr(dataAsStrRLE)
			cast('ast.Call', Ima___Assign.value).args = [dataAs_ast_expr]
		case _:
			pass
	return Ima___Assign

def makeJob(job: RecipeJobTheorem2) -> None:
	"""Generate an optimized module for map folding calculations.

	This function orchestrates the complete code transformation assembly line to convert a generic map folding algorithm into a
	highly optimized, specialized computation module.

	Parameters
	----------
	job : RecipeJobTheorem2
		Configuration recipe containing source locations, target paths, and state.

	"""
	ingredientsCount: IngredientsFunction = IngredientsFunction(raiseIfNone(extractFunctionDef(job.source_astModule, job.countCallable)))
	ingredientsCount.astFunctionDef.decorator_list = []

	# Replace identifiers-with-static-values with their values.
	listIdentifiersStaticValues: list[str] = listIdentifiersStaticValuesHARDCODED
	for identifier in listIdentifiersStaticValues:
		NodeChanger(IfThis.isNameIdentifier(identifier)
			, Then.replaceWith(Make.Constant(int(job.state.__dict__[identifier])))
		).visit(ingredientsCount.astFunctionDef)

	ingredientsCount.imports.update(job.shatteredDataclass.imports)
	ingredientsCount = removeUnusedParameters(ingredientsCount)
	NodeChanger(Be.arg, lambda removeIt: ingredientsCount.astFunctionDef.body.insert(0, _move_arg2body(removeIt.arg, job))).visit(ingredientsCount.astFunctionDef)

	ingredientsCount = _addWriteFoldsTotal(ingredientsCount, job)
	ingredientsCount = _variableCompatibility(ingredientsCount, job)

	ingredientsModule = IngredientsModule(launcher=Make.Module([
		Make.If(Make.Compare(Make.Name('__name__'), [Make.Eq()], [Make.Constant('__main__')])
			, body=[Make.Expr(Make.Call(Make.Name(job.countCallable)))])]))

	ingredientsCount, ingredientsModule = _datatypeDefinitions(ingredientsCount, ingredientsModule)

	ingredientsModule.appendIngredientsFunction(ingredientsCount)

	if sys.platform == 'linux':
		buildCommand: list[str] = ['codon', 'build', '--exe', '--release',
			'--fast-math', '--enable-unsafe-fp-math', '--disable-exceptions',
			'--mcpu=native',
			'-o', str(job.pathFilenameModule.with_suffix('')),
			'-']
		streamText = subprocess.Popen(buildCommand, stdin=subprocess.PIPE, text=True)
		if streamText.stdin is not None:
			write_astModule(ingredientsModule, pathFilename=cast('TextIOBase', streamText.stdin), packageName=job.packageIdentifier)
			streamText.stdin.close()
		streamText.wait()
		subprocess.run(['/usr/bin/strip', str(job.pathFilenameModule.with_suffix(''))], check=False)
		sys.stdout.write(f"sudo systemd-run --unit={job.moduleIdentifier} --nice=-10 --property=CPUAffinity=0 {job.pathFilenameModule.with_suffix('')}\n")
	else:
		write_astModule(ingredientsModule, pathFilename=job.pathFilenameModule, packageName=job.packageIdentifier)

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
	state = initializeGroupsOfFolds(MapFoldingState(mapShape))
	pathModule = PurePosixPath(Path.home(), 'mapFolding', 'jobs')
	pathFilenameFoldsTotal = PurePosixPath(getPathFilenameFoldsTotal(state.mapShape, pathModule))
	aJob = RecipeJobTheorem2(state, pathModule=pathModule, pathFilenameFoldsTotal=pathFilenameFoldsTotal)
	makeJob(aJob)

if __name__ == '__main__':
	mapShape = (2, 21)
	fromMapShape(mapShape)

