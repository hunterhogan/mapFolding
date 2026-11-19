"""Generate axis-meander modules with reflection or NumPy counting.

(AI generated docstring)

You can transform algorithm syntax trees into variants that combine counts for reflected arch
codes or count axis-meanders with NumPy arrays. The transformations modify the supplied syntax
trees. `makeModulesAxis` reads the source modules and writes the three generated variants.

Contents
--------
Functions
	makeAxisPileNumPy
		Combine NumPy counting with pile-axis completion.
	makeAxisPileOrbit
		Combine reflection-orbit counting with pile-axis completion.
	makeModulesAxis
		Write the axis-meander orbit modules and the pile-axis NumPy module.
	makeOrbit
		Represent each generated arch code by the smaller code in its reflection orbit.
Variables
	defaultMatrixAxis
		Access identifier and filesystem defaults for axis-meander generation.
"""

from __future__ import annotations

from astToolkit import Be, DOT, Grab, Make, NodeChanger, NodeTourist, Then
from astToolkit.filesystem import parsePathFilename2astModule
from copy import deepcopy
from hunterMakesPy import raiseIfNone
from mapFolding.kitAST import IfThis, IngredientsModule, removeFunctionDef, toDisk
from mapFolding.kitAST.paths import getLogicalPath, getPathFilename
from mapFolding.kitAST.theSSOT import defaultMatrixMeanders
from typing import TYPE_CHECKING
import ast

if TYPE_CHECKING:
	from mapFolding.theTypes import Default
	from pathlib import PurePath
	from typing import Any

defaultMatrixAxis: Default = deepcopy(defaultMatrixMeanders)
"""Access identifier and filesystem defaults for axis-meander generation.

(AI generated docstring)

This independent copy of `mapFolding.kitAST.theSSOT.defaultMatrixMeanders` adds reflection,
axis-module, and analyzed-arch-code identifiers. The mapping remains mutable.
"""
defaultMatrixAxis['function'].update({'reflect': 'reflectArcCode'})
defaultMatrixAxis['module'].update({
	'axisPile': 'matrixAxisPile'
	, 'axisPileNumPy': 'matrixAxisPileNumPy'
	, 'axisPileOrbit': 'matrixAxisPileOrbit'
	, 'axisPileDeadEnd': 'matrixAxisPileDeadEnd'
	, 'axisPileDeadEndOrbit': 'matrixAxisPileDeadEndOrbit'
	, 'axisStrip': 'matrixAxisStrip'
	, 'axisStripOrbit': 'matrixAxisStripOrbit'
	, 'axisStripGap': 'matrixAxisStripGap'
	, 'axisStripGapOrbit': 'matrixAxisStripGapOrbit'
})
defaultMatrixAxis['variable'].update({'arcCodeAnalysis': 'arcCodeAnalysis'})

def makeOrbit(astModule: ast.Module, identifiers: Default | None = None, **override: Any) -> ast.Module:
	"""Combine reflected arch codes under one representative in a counting module.

	(AI generated docstring)

	You can make counting use the smaller of each analyzed arch code and its reflection before
	accumulating counts. The supplied counting function must compare the analyzed arch code as the
	left operand of its accumulation conditions. The transformed function requires the reflection
	function in its module namespace when executed.

	Parameters
	----------
	astModule : ast.Module
		Counting module to transform.
	identifiers : Default = defaultMatrixAxis
		Identifier mapping. `None` or an empty mapping uses `defaultMatrixAxis`.
	**override : Any
		Nonempty `名Callable`, `名CallableReflect`, and `名ArcCodeAnalysis` values replace the
		mapping's counting-function, reflection-function, and analyzed-arch-code identifiers.
		Other entries have no effect.

	Returns
	-------
	astModule : ast.Module
		The supplied module with reflection representatives assigned before accumulation.

	See Also
	--------
	`makeAxisPileOrbit`
		Combine reflection-orbit counting with the pile-axis completion algorithm.

	Input Mutation
	--------------
	The supplied syntax tree is modified in place. Repeating the transformation inserts additional
	assignments. The reflection function remains a dependency of the transformed counting function.
	"""
	identifiers = identifiers or defaultMatrixAxis
	名Callable: str = override.get('名Callable') or identifiers['function']['counting']
	名CallableReflect: str = override.get('名CallableReflect') or identifiers['function']['reflect']
	名ArcCodeAnalysis: str = override.get('名ArcCodeAnalysis') or identifiers['variable']['arcCodeAnalysis']

	astAssignOrbit: ast.Assign = Make.Assign([Make.Name(名ArcCodeAnalysis, Make.Store())], Make.Call(Make.Name('min')
		, [Make.Name(名ArcCodeAnalysis), Make.Call(Make.Name(名CallableReflect), [Make.Name(名ArcCodeAnalysis)])]))
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(名Callable))
		, NodeChanger(Be.If.testIs(Be.Compare.leftIs(IfThis.isNameIdentifier(名ArcCodeAnalysis)))
			, Grab.bodyAttribute(Grab.index(0, Then.insertThisAbove([astAssignOrbit])))).visit).visit(astModule)
	return astModule

def makeAxisPileOrbit(astModuleAxisPile: ast.Module, astModuleCounting: ast.Module, identifiers: Default | None = None, **override: Any
	) -> ast.Module:
	"""Combine reflection-orbit counting with pile-axis completion.

	(AI generated docstring)

	You can assemble the counting definitions and pile-axis definitions into one module. Counting
	uses the smaller arch code in each reflection orbit, and the pile-axis dispatcher completes
	the count.

	Parameters
	----------
	astModuleAxisPile : ast.Module
		Pile-axis module containing its dispatcher, reflection function, and counting import.
	astModuleCounting : ast.Module
		Counting module whose dispatcher will be removed.
	identifiers : Default = defaultMatrixAxis
		Identifier mapping. `None` or an empty mapping uses `defaultMatrixAxis`.
	**override : Any
		Nonempty `名CallableDispatcher`, `package`, `logicalPathAlgorithm`, and `名ModuleAlgorithm`
		values select the dispatcher and counting import to remove. The identifier overrides
		accepted by `makeOrbit` also apply.

	Returns
	-------
	astModule : ast.Module
		New module containing transformed counting definitions followed by pile-axis definitions.

	Input Mutation
	--------------
	Both supplied syntax trees are modified. The counting dispatcher and the pile-axis counting
	import are removed, and `makeOrbit` modifies the counting function. The returned module shares
	the retained nodes with the supplied modules.
	"""
	identifiers = identifiers or defaultMatrixAxis
	名CallableDispatcher: str = override.get('名CallableDispatcher') or identifiers['function']['dispatcher']
	名Package: str = override.get('package') or identifiers['module']['package']
	logicalPathAlgorithm: str = override.get('logicalPathAlgorithm') or identifiers['logicalPath']['algorithm']
	名ModuleAlgorithm: str = override.get('名ModuleAlgorithm') or identifiers['module']['algorithm']

	NodeChanger(Be.ImportFrom.moduleIs(IfThis.isIdentifier(getLogicalPath(名Package, logicalPathAlgorithm, 名ModuleAlgorithm)))
		, Then.removeIt).visit(astModuleAxisPile)
	removeFunctionDef(名CallableDispatcher, astModuleCounting)
	return Make.Module([*makeOrbit(astModuleCounting, identifiers, **override).body, *astModuleAxisPile.body])

def makeAxisPileNumPy(astModuleAxisPile: ast.Module, astModuleNumPy: ast.Module, identifiers: Default | None = None, **override: Any
	) -> ast.Module:
	"""Combine NumPy meander counting with pile-axis completion.

	(AI generated docstring)

	You can add pile-axis completion to a NumPy counting module. The NumPy dispatcher discards
	zero-count lookup entries before applying the pile-axis completion statements. Boundary progress
	uses a text postfix compatible with the NumPy counting state.

	Parameters
	----------
	astModuleAxisPile : ast.Module
		Pile-axis module whose dispatcher begins with counting, ends with a return, and contains
		the completion statements between them.
	astModuleNumPy : ast.Module
		NumPy counting module with a dispatcher that returns the completed state.
	identifiers : Default = defaultMatrixAxis
		Identifier mapping. `None` or an empty mapping uses `defaultMatrixAxis`.
	**override : Any
		Nonempty `名CallableDispatcher`, `package`, `logicalPathAlgorithm`, and `名ModuleAlgorithm`
		values select the dispatcher and counting import. Nonempty `名DataclassInstance` and
		`名TqdmBoundary` values select the state and progress-display identifiers.

	Returns
	-------
	astModule : ast.Module
		New module containing the `itemgetter` import, transformed NumPy definitions, and remaining
		pile-axis definitions.

	Input Mutation
	--------------
	Both supplied syntax trees are modified. The pile-axis counting import and dispatcher are
	removed. The NumPy progress calls and dispatcher are modified. The returned module shares the
	retained nodes with the supplied modules.
	"""
	identifiers = identifiers or defaultMatrixAxis
	名CallableDispatcher: str = override.get('名CallableDispatcher') or identifiers['function']['dispatcher']
	名Package: str = override.get('package') or identifiers['module']['package']
	logicalPathAlgorithm: str = override.get('logicalPathAlgorithm') or identifiers['logicalPath']['algorithm']
	名ModuleAlgorithm: str = override.get('名ModuleAlgorithm') or identifiers['module']['algorithm']

	NodeChanger(Be.ImportFrom.moduleIs(IfThis.isIdentifier(getLogicalPath(名Package, logicalPathAlgorithm, 名ModuleAlgorithm)))
		, Then.removeIt).visit(astModuleAxisPile)
	boxOfStatementsJoined: list[ast.stmt] = raiseIfNone(NodeTourist[ast.FunctionDef, list[ast.stmt]](
		Be.FunctionDef.nameIs(IfThis.isIdentifier(名CallableDispatcher)), Then.extractIt(DOT.body)).captureLastMatch(astModuleAxisPile))[1:-1]
	removeFunctionDef(名CallableDispatcher, astModuleAxisPile)

	名DataclassInstance: str = override.get('名DataclassInstance') or identifiers['variable']['stateInstance']
	名TqdmBoundary: str = override.get('名TqdmBoundary') or identifiers['variable']['tqdmBoundary']
	NodeChanger(Be.Call.funcIs(IfThis.isAttributeNamespaceIdentifier(名TqdmBoundary, 'set_postfix'))
		, Then.replaceWith(Make.Call(Make.Attribute(Make.Name(名TqdmBoundary), 'set_postfix_str'), [
			Make.JoinedStr([Make.Constant('boundary='), Make.FormattedValue(Make.Attribute(Make.Name(名DataclassInstance), 'boundary'), conversion=-1)])]))
		).visit(astModuleNumPy)
	astAttributeLookup: ast.Attribute = Make.Attribute(Make.Name(名DataclassInstance), 'lookupMeanders')
	astAssignStates: ast.Assign = Make.Assign([Make.Attribute(Make.Name(名DataclassInstance), 'lookupMeanders', context=Make.Store())]
		, Make.Call(Make.Name('dict'), [Make.Call(Make.Name('filter'), [Make.Call(Make.Name('itemgetter'), [Make.Constant(1)])
			, Make.Call(Make.Attribute(astAttributeLookup, 'items'))])]))
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(名CallableDispatcher))
		, NodeChanger(Be.Return, Then.insertThisAbove([astAssignStates, *boxOfStatementsJoined])).visit).visit(astModuleNumPy)
	return Make.Module([Make.ImportFrom('operator', [Make.alias('itemgetter')]), *astModuleNumPy.body, *astModuleAxisPile.body])

def makeModulesAxis() -> None:
	"""Write the axis-meander orbit modules and the pile-axis NumPy module.

	(AI generated docstring)

	You can generate all five variants using the current `defaultMatrixAxis` settings. Source
	algorithms are read from the configured source package and algorithm path. Imports are
	consolidated before each generated module is written.

	See Also
	--------
	`makeOrbit`
		Transform strip-axis, strip-gap, and dead-end pile counting to use reflection representatives.
	`makeAxisPileOrbit`
		Assemble pile-axis reflection-orbit counting.
	`makeAxisPileNumPy`
		Assemble pile-axis NumPy counting.

	File Replacement
	----------------
	The configured `axisStripOrbit`, `axisStripGapOrbit`, `axisPileDeadEndOrbit`, `axisPileOrbit`, and
	`axisPileNumPy` output modules are written to the configured generated-module directory, replacing
	existing files.
	"""
	identifiers: Default = defaultMatrixAxis
	pathRoot: PurePath = identifiers['filesystem']['sourcePackage']
	logicalPathAlgorithm: str = identifiers['logicalPath']['algorithm']

	pathFilenameAxisPile: PurePath = getPathFilename(pathRoot, logicalPathAlgorithm, identifiers['module']['axisPile'])
	pathFilenameAxisStrip: PurePath = getPathFilename(pathRoot, logicalPathAlgorithm, identifiers['module']['axisStrip'])

	ingredientsModule = IngredientsModule(epilogue=makeOrbit(parsePathFilename2astModule(pathFilenameAxisStrip), identifiers))
	ingredientsModule.imports.extractImports(ingredientsModule.epilogue)
	toDisk(ingredientsModule, identifiers, {}, identifiers['module']['axisStripOrbit'])

	ingredientsModule = IngredientsModule(epilogue=makeOrbit(parsePathFilename2astModule(getPathFilename(
		pathRoot, logicalPathAlgorithm, identifiers['module']['axisStripGap'])), identifiers))
	ingredientsModule.imports.extractImports(ingredientsModule.epilogue)
	toDisk(ingredientsModule, identifiers, {}, identifiers['module']['axisStripGapOrbit'])

	ingredientsModule = IngredientsModule(epilogue=makeOrbit(parsePathFilename2astModule(getPathFilename(
		pathRoot, logicalPathAlgorithm, identifiers['module']['axisPileDeadEnd'])), identifiers))
	ingredientsModule.imports.extractImports(ingredientsModule.epilogue)
	toDisk(ingredientsModule, identifiers, {}, identifiers['module']['axisPileDeadEndOrbit'])

	ingredientsModule = IngredientsModule(epilogue=makeAxisPileOrbit(parsePathFilename2astModule(pathFilenameAxisPile)
		, parsePathFilename2astModule(getPathFilename(pathRoot, logicalPathAlgorithm, identifiers['module']['algorithm'])), identifiers))
	ingredientsModule.imports.extractImports(ingredientsModule.epilogue)
	toDisk(ingredientsModule, identifiers, {}, identifiers['module']['axisPileOrbit'])

	ingredientsModule = IngredientsModule(epilogue=makeAxisPileNumPy(parsePathFilename2astModule(pathFilenameAxisPile)
		, parsePathFilename2astModule(getPathFilename(pathRoot, logicalPathAlgorithm, identifiers['module']['numpy'])), identifiers))
	ingredientsModule.imports.extractImports(ingredientsModule.epilogue)
	toDisk(ingredientsModule, identifiers, {}, identifiers['module']['axisPileNumPy'])

if __name__ == '__main__':
	makeModulesAxis()
