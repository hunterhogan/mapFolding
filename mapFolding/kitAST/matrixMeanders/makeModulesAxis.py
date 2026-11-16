from __future__ import annotations

from astToolkit import Be, DOT, Grab, Make, NodeChanger, NodeTourist, Then
from astToolkit.filesystem import parsePathFilename2astModule
from copy import deepcopy
from hunterMakesPy import raiseIfNone
from mapFolding.kitAST import IfThis, removeFunctionDef, toDisk
from mapFolding.kitAST.matrixMeanders.makeModules import combineModules
from mapFolding.kitAST.paths import getLogicalPath, getPathFilename
from mapFolding.kitAST.theSSOT import defaultMatrixMeanders
from typing import TYPE_CHECKING
import ast

if TYPE_CHECKING:
	from mapFolding.theTypes import Default
	from pathlib import PurePath
	from typing import Any

defaultMatrixAxis: Default = deepcopy(defaultMatrixMeanders)
defaultMatrixAxis['function'].update({'reflect': 'reflectArcCode'})
defaultMatrixAxis['module'].update({
	'axisPile': 'matrixAxisPile'
	, 'axisPileNumPy': 'matrixAxisPileNumPy'
	, 'axisPileOrbit': 'matrixAxisPileOrbit'
	, 'axisStrip': 'matrixAxisStrip'
	, 'axisStripOrbit': 'matrixAxisStripOrbit'
})
defaultMatrixAxis['variable'].update({'arcCodeAnalysis': 'arcCodeAnalysis'})

def makeOrbit(astModule: ast.Module, identifiers: Default | None = None, **override: Any) -> ast.Module:
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
	identifiers: Default = defaultMatrixAxis
	pathRoot: PurePath = identifiers['filesystem']['sourcePackage']
	logicalPathAlgorithm: str = identifiers['logicalPath']['algorithm']

	pathFilenameAxisPile: PurePath = getPathFilename(pathRoot, logicalPathAlgorithm, identifiers['module']['axisPile'])
	pathFilenameAxisStrip: PurePath = getPathFilename(pathRoot, logicalPathAlgorithm, identifiers['module']['axisStrip'])

	toDisk(combineModules(makeOrbit(parsePathFilename2astModule(pathFilenameAxisStrip), identifiers), Make.Module([]))
		, identifiers, {}, identifiers['module']['axisStripOrbit'])

	toDisk(combineModules(makeAxisPileOrbit(parsePathFilename2astModule(pathFilenameAxisPile), parsePathFilename2astModule(
		getPathFilename(pathRoot, logicalPathAlgorithm, identifiers['module']['algorithm'])), identifiers), Make.Module([]))
		, identifiers, {}, identifiers['module']['axisPileOrbit'])

	toDisk(combineModules(makeAxisPileNumPy(parsePathFilename2astModule(pathFilenameAxisPile), parsePathFilename2astModule(
		getPathFilename(pathRoot, logicalPathAlgorithm, identifiers['module']['numpy'])), identifiers), Make.Module([]))
		, identifiers, {}, identifiers['module']['axisPileNumPy'])

if __name__ == '__main__':
	makeModulesAxis()
