"""codon.

https://docs.exaloop.io/start/install/"""

from astToolkit import (
	Be, DOT, extractFunctionDef, Grab, hasDOTvalue, identifierDotAttribute, IngredientsFunction, IngredientsModule, Make,
	NodeChanger, NodeTourist, Then)
from astToolkit.transformationTools import write_astModule
from hunterMakesPy import autoDecodingRLE, raiseIfNone
from mapFolding import getPathFilenameFoldsTotal, MapFoldingState, packageSettings
from mapFolding.someAssemblyRequired import IfThis
from mapFolding.someAssemblyRequired.RecipeJob import RecipeJobTheorem2
from mapFolding.syntheticModules.initializeCount import initializeGroupsOfFolds
from pathlib import Path, PurePosixPath
from typing import cast, NamedTuple
import ast
import subprocess
import sys

class DatatypeConfiguration(NamedTuple):
	"""Configuration for mapping framework datatypes to compiled datatypes.

	This configuration class defines how abstract datatypes used in the map folding framework should be replaced with compiled
	datatypes during code generation. Each configuration specifies the source module, target type name, and optional import
	alias for the transformation.

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

listDatatypeConfigs: list[DatatypeConfiguration] = [
	DatatypeConfiguration(datatypeIdentifier='DatatypeLeavesTotal', typeModule='numpy', typeIdentifier='uint16', type_asname='DatatypeLeavesTotal'),
	DatatypeConfiguration(datatypeIdentifier='DatatypeElephino', typeModule='numpy', typeIdentifier='uint16', type_asname='DatatypeElephino'),
	DatatypeConfiguration(datatypeIdentifier='DatatypeFoldsTotal', typeModule='numpy', typeIdentifier='int64', type_asname='DatatypeFoldsTotal'),
]

listNumPyTypeConfigs: list[DatatypeConfiguration] = [
	DatatypeConfiguration(datatypeIdentifier='Array1DLeavesTotal', typeModule='numpy', typeIdentifier='uint16', type_asname='Array1DLeavesTotal'),
	DatatypeConfiguration(datatypeIdentifier='Array1DElephino', typeModule='numpy', typeIdentifier='uint16', type_asname='Array1DElephino'),
	DatatypeConfiguration(datatypeIdentifier='Array3D', typeModule='numpy', typeIdentifier='uint16', type_asname='Array3D'),
]

def _datatypeDefinitions(ingredientsFunction: IngredientsFunction, ingredientsModule: IngredientsModule) -> tuple[IngredientsFunction, IngredientsModule]:
	for datatypeConfig in listDatatypeConfigs:
		ingredientsFunction.imports.removeImportFrom(datatypeConfig.typeModule, None, datatypeConfig.datatypeIdentifier)
		ingredientsFunction.imports.addImportFrom_asStr(datatypeConfig.typeModule, datatypeConfig.typeIdentifier, datatypeConfig.type_asname)
		continue
		ingredientsModule.appendPrologue(statement=Make.Assign([Make.Name(datatypeConfig.datatypeIdentifier, ast.Store())]
			, value=Make.Name(datatypeConfig.typeIdentifier)
		))

	for datatypeConfig in listNumPyTypeConfigs:
		ingredientsFunction.imports.removeImportFrom(datatypeConfig.typeModule, None, datatypeConfig.datatypeIdentifier)
		ingredientsFunction.imports.addImportFrom_asStr(datatypeConfig.typeModule, datatypeConfig.typeIdentifier, datatypeConfig.type_asname)
		continue
		ingredientsModule.appendPrologue(statement=Make.Assign([Make.Name(datatypeConfig.datatypeIdentifier, ast.Store())]
			, value=Make.Name(datatypeConfig.typeIdentifier)
		))

	ingredientsFunction.imports.removeImportFromModule('mapFolding.dataBaskets')

	return ingredientsFunction, ingredientsModule

def _pythonCode2expr(string: str) -> ast.expr:
	"""Convert *one* expression as a string of Python code to an `ast.expr`."""
	return raiseIfNone(NodeTourist(Be.Expr, Then.extractIt(DOT.value)).captureLastMatch(ast.parse(string)))

def _variableCompatibility(ingredientsFunction: IngredientsFunction, job: RecipeJobTheorem2) -> IngredientsFunction:
	for ast_arg in job.shatteredDataclass.list_argAnnotated4ArgumentsSpecification:
		identifier = ast_arg.arg
		annotation = raiseIfNone(ast_arg.annotation)

		# `identifier` in Augmented Assignment.
		NodeChanger(Be.AugAssign.targetIs(IfThis.isNestedNameIdentifier(identifier))
			, doThat=lambda node: Grab.valueAttribute(Then.replaceWith(Make.Call(annotation, listParameters=[node.value])))(node)
		).visit(ingredientsFunction.astFunctionDef)

		# `identifier` in Assignments; exclude `numpy.array`.
		NodeChanger(findThis=lambda node: IfThis.isAssignAndTargets0Is(IfThis.isNameIdentifier(identifier))(node)
			and Be.Assign.valueIs(Be.Constant)(node)
			, doThat=lambda node: Grab.valueAttribute(Then.replaceWith(Make.Call(annotation, listParameters=[node.value])))(node)
		).visit(ingredientsFunction.astFunctionDef)

		# `identifier` - 1.
		NodeChanger(Be.BinOp.leftIs(IfThis.isNestedNameIdentifier(identifier))
			, doThat=lambda node: Grab.rightAttribute(Then.replaceWith(Make.Call(annotation, listParameters=[node.right])))(node)
		).visit(ingredientsFunction.astFunctionDef)

		# `identifier` in Comparison.
		NodeChanger(Be.Compare.leftIs(IfThis.isNestedNameIdentifier(identifier))
			, doThat=lambda node: Grab.comparatorsAttribute(lambda at: Then.replaceWith([Make.Call(annotation, listParameters=[node.comparators[0]])])(at[0]))(node)
		).visit(ingredientsFunction.astFunctionDef)

		# `identifier` has exactly one index value.
		NodeChanger(
			findThis=lambda node: Be.Subscript.valueIs(IfThis.isNestedNameIdentifier(identifier))(node)
			and not Be.Subscript.sliceIs(Be.Tuple)(node)
			, doThat=lambda node: Grab.sliceAttribute(Then.replaceWith(Make.Call(Make.Name('int'), listParameters=[node.slice])))(node)
		).visit(ingredientsFunction.astFunctionDef)

		# `identifier` has multiple index values.
		NodeChanger(
			findThis=lambda node: Be.Subscript.valueIs(IfThis.isNestedNameIdentifier(identifier))(node)
			and Be.Subscript.sliceIs(Be.Tuple)(node)
			, doThat=lambda node: Grab.sliceAttribute(Grab.eltsAttribute(
				Then.replaceWith([
					Make.Call(Make.Name('int'), listParameters=[cast(ast.Tuple, node.slice).elts[index]])
					for index in range(len(cast(ast.Tuple, node.slice).elts))
					])
			))(node)
		).visit(ingredientsFunction.astFunctionDef)

	return ingredientsFunction

def move_arg2FunctionDefDOTbodyAndAssignInitialValues(ingredientsFunction: IngredientsFunction, job: RecipeJobTheorem2) -> IngredientsFunction:
	"""Convert function parameters into initialized variables with concrete values.

	This function converts function arguments into statically initialized variables in the function body.

	The function handles different data types (scalars, arrays, custom types) appropriately, replacing abstract parameter
	references with concrete values from the computation state. It also removes unused parameters and variables.

	Parameters
	----------
	ingredientsFunction : IngredientsFunction
		The function to transform.
	job : RecipeJobTheorem2
		Recipe containing concrete values for parameters and field metadata.

	Returns
	-------
	modifiedFunction : IngredientsFunction
		The modified function with parameters converted to initialized variables.

	"""
	ingredientsFunction.imports.update(job.shatteredDataclass.imports)

	list_argCuzMyBrainRefusesToThink: list[ast.arg] = ingredientsFunction.astFunctionDef.args.args + ingredientsFunction.astFunctionDef.args.posonlyargs + ingredientsFunction.astFunctionDef.args.kwonlyargs
	list_arg_arg: list[str] = [ast_arg.arg for ast_arg in list_argCuzMyBrainRefusesToThink]
	listName: list[ast.Name] = []
	NodeTourist(Be.Name, Then.appendTo(listName)).visit(ingredientsFunction.astFunctionDef)
	listIdentifiers: list[str] = [astName.id for astName in listName]
	listIdentifiersNotUsed: list[str] = list(set(list_arg_arg) - set(listIdentifiers))

	for ast_arg in list_argCuzMyBrainRefusesToThink:
		if ast_arg.arg in job.shatteredDataclass.field2AnnAssign:
			if ast_arg.arg in listIdentifiersNotUsed:
				pass
			else:
				Ima___Assign, elementConstructor = job.shatteredDataclass.Z0Z_field2AnnAssign[ast_arg.arg]
				match elementConstructor:
					case 'scalar':
						cast('ast.Constant', cast('ast.Call', Ima___Assign.value).args[0]).value = int(job.state.__dict__[ast_arg.arg])
					case 'array':
						dataAsStrRLE: str = autoDecodingRLE(job.state.__dict__[ast_arg.arg], assumeAddSpaces=True)
						dataAs_ast_expr: ast.expr = _pythonCode2expr(dataAsStrRLE)
						cast('ast.Call', Ima___Assign.value).args = [dataAs_ast_expr]
					case _:
						pass

				ingredientsFunction.astFunctionDef.body.insert(0, Ima___Assign)

			NodeChanger(IfThis.is_argIdentifier(ast_arg.arg), Then.removeIt).visit(ingredientsFunction.astFunctionDef)

	ast.fix_missing_locations(ingredientsFunction.astFunctionDef)
	return ingredientsFunction

def makeJob(job: RecipeJobTheorem2) -> None:
	"""Generate an optimized module for map folding calculations.

	This function orchestrates the complete code transformation assembly line to convert
	a generic map folding algorithm into a highly optimized, specialized computation
	module.

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

	linesLaunch: str = f"""
if __name__ == '__main__':
	foldsTotal = {job.countCallable}()
	print('\\nmap {job.state.mapShape} =', foldsTotal)
	writeStream = open('{job.pathFilenameFoldsTotal.as_posix()}', 'w')
	writeStream.write(str(foldsTotal))
	writeStream.close()
"""
	ingredientsModule = IngredientsModule(launcher=ast.parse(linesLaunch))

	# TODO think about `groupsOfFolds *= DatatypeFoldsTotal({2 * job.state.leavesTotal}); return groupsOfFolds`
	NodeChanger(Be.Return
		, Then.replaceWith(Make.Return(job.shatteredDataclass.countingVariableName))).visit(ingredientsCount.astFunctionDef)
	groupsOfFolds2foldsTotal = NodeChanger[ast.AugAssign, hasDOTvalue](
		findThis=(lambda node: Be.AugAssign.targetIs(IfThis.isNameIdentifier(job.shatteredDataclass.countingVariableName.id))(node)
			and Be.AugAssign.opIs(Be.Mult)(node)
			and Be.AugAssign.valueIs(Be.Constant)(node)
		)
		, doThat=lambda node: Grab.valueAttribute(Then.replaceWith(Make.Constant(job.state.leavesTotal * ast.literal_eval(node.value))))(node)
	)
	groupsOfFolds2foldsTotal.visit(ingredientsCount.astFunctionDef)

	# TODO think about assigning `returns` here, then removing `returns` a few lines from now.
	ingredientsCount.astFunctionDef.returns = job.shatteredDataclass.countingVariableAnnotation

	ingredientsCount = move_arg2FunctionDefDOTbodyAndAssignInitialValues(ingredientsCount, job)

	ingredientsCount, ingredientsModule = _datatypeDefinitions(ingredientsCount, ingredientsModule)
	# NOTE Reminder, `returns = None` means "type information is null"; the identifier for the return was removed by `_datatypeDefinitions`.
	ingredientsCount.astFunctionDef.returns = None

	ingredientsCount = _variableCompatibility(ingredientsCount, job)

	ingredientsModule.appendIngredientsFunction(ingredientsCount)
	write_astModule(ingredientsModule, pathFilename=job.pathFilenameModule, packageName=job.packageIdentifier)

	if sys.platform == 'linux':
		pathFilenameBuild = Path.home() / 'mapFolding' / 'jobs' / job.pathFilenameModule.stem
		pathFilenameBuild.parent.mkdir(parents=True, exist_ok=True)

		buildCommand = ['codon', 'build', '-release', '-disable-exceptions', '-o', str(pathFilenameBuild), str(job.pathFilenameModule)]
		subprocess.run(buildCommand)

if __name__ == '__main__':
	state = initializeGroupsOfFolds(MapFoldingState((2,4)))
	pathModule = PurePosixPath(packageSettings.pathPackage, 'jobs')
	# TODO put `pathFilenameFoldsTotal` in wsl.
	pathFilenameFoldsTotal = PurePosixPath(getPathFilenameFoldsTotal(state.mapShape, pathModule))
	aJob = RecipeJobTheorem2(state, pathModule=pathModule, pathFilenameFoldsTotal=pathFilenameFoldsTotal)
	makeJob(aJob)
