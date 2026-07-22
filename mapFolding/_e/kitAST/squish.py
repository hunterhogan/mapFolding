from __future__ import annotations

from anyascii import anyascii
from ast import parse as ast_parse
from astToolkit import Be, IfThis, Make, NodeChanger, NodeTourist, parseLogicalPath2astModule, Then
from astToolkit.containers import astModuleToIngredientsFunction, IngredientsModule, LedgerOfImports
from humpy_cytoolz import juxt
from itertools import repeat
from mapFolding import packageSettings
from mapFolding._e.kitAST.infoBooth import default
from pathlib import Path
from typing import TYPE_CHECKING
import autoflake  # pyright: ignore[reportMissingTypeStubs]
import python_minifier
import subprocess  # ruff:ignore[suspicious-subprocess-import]
import sys

if TYPE_CHECKING:
	from collections.abc import Callable
	from hunterMakesPy import identifierDotAttribute
	from typing import TypeIs
	import ast

launcher = """
# ruff:file-ignore[commented-out-code, print]
if __name__ == "__main__":
	CPUlimit: int | float | None = None
	state: EliminationState = EliminationState((2,) * 5)
	# state = pinPile零Ante首零(state)
	state = pinPilesAtEnds(state, 3)
	state = pinLeavesDimension首二(state)
	# state = pin3beans2(state)
	# state = pin首beans(state)
	# state = pinLeavesDimension一(state)
	# state = pinLeavesDimension二(state)
	state = pinLeavesDimensions0零一(state)
	workersMaximum: int = defineProcessorLimit(CPUlimit)
	print(doTheNeedful(state, workersMaximum).foldsTotal)

"""

ledgerTYPE_CHECKING = LedgerOfImports()
module1 = IngredientsModule()

def _dissectModule(logicalPath: identifierDotAttribute) -> ast.Module:
	moduleDissect: ast.Module = parseLogicalPath2astModule(logicalPath, optimize=2)

	findThis: Callable[[ast.AST], TypeIs[ast.If]] = Be.If.testIs(Be.Name.idIs('TYPE_CHECKING'.__eq__))
	NodeTourist(findThis, ledgerTYPE_CHECKING.walkThis).visit(moduleDissect)
	NodeChanger(findThis, Then.removeIt).visit(moduleDissect)

	return moduleDissect

def assimilateFunction(logicalPath: identifierDotAttribute, identifierFunction: str) -> None:
	module1.appendIngredientsFunction(astModuleToIngredientsFunction(_dissectModule(logicalPath), identifierFunction))

def assimilateModule(logicalPath: identifierDotAttribute) -> None:
	moduleDissect: ast.Module = _dissectModule(logicalPath)

	module1.imports.walkThis(moduleDissect)
	NodeChanger(IfThis.isAnyOf(Be.Import, Be.ImportFrom), Then.removeIt).visit(moduleDissect)

	NodeChanger(Be.Expr.valueIs(Be.Constant.valueIs(lambda fu: isinstance(fu, str))), Then.removeIt).visit(moduleDissect)

	module1.appendEpilogue(moduleDissect)

def assimilateEliminationCrease(identifierModule: str) -> Path:
	pathFilename = Path(*default['logicalPath']['synthetic'].split('.'), identifierModule + packageSettings.fileExtension)

	listModules: list[identifierDotAttribute] = [
		*tuple(map("{0}._e.{1}".format, repeat(packageSettings.identifierPackage), (
			'theTypes', 'semiotics', 'leafDomains', 'pileOptions', '_disaggregation', '_beDRY', 'dataBaskets', 'filters', 'pinIt'
		)))
		, *tuple(map("{0}._e._2上nDimensional.{1}".format, repeat(packageSettings.identifierPackage), (
			'pinIt', 'pinByCrease', 'pinByDomain', 'pinItAnnex', 'semiotics', 'beDRY', 'measure', 'creases'
			, 'leafDomains', 'conditionalOrdering', 'pileOptions', 'filters'
		)))
		, f"{default['logicalPath']['algorithm']}.iff"
		, f"{default['logicalPath']['algorithm']}.{default['module']['algorithm']}"
	]

	listPackages: list[identifierDotAttribute] = [
		*tuple(map("{0}.{1}".format, repeat(packageSettings.identifierPackage), ('beDRY', '_e', '_e._2上nDimensional')))
		, *listModules
	]

	assimilateFunction('mapFolding.beDRY', 'getLeavesTotal')
	assimilateFunction('mapFolding.beDRY', 'defineProcessorLimit')
	tuple(map(assimilateModule, listModules))

	tuple(map(juxt(ledgerTYPE_CHECKING.removeImportFrom, module1.removeImportFrom), listPackages, repeat(None)))

	module1.appendPrologue(statement=Make.If(Make.Name('TYPE_CHECKING'), ledgerTYPE_CHECKING.makeList_ast()))
	module1.appendLauncher(ast_parse(launcher))

	return module1.write_astModule(pathFilename, packageSettings.identifierPackage)

def minify(pathFilename: Path) -> Path:
	pathFilename.with_stem('min').write_text(python_minifier.minify(autoflake.fix_code(pathFilename.read_text(encoding='utf-8'), remove_unused_variables=True)
		, remove_literal_statements=True, rename_globals=True, prefer_single_line=False), encoding='utf-8')
	return pathFilename.with_stem('min')

def toASCII(pathFilename: Path) -> Path:
	pathFilename.with_stem('ascii').write_text(anyascii(pathFilename.read_text(encoding='utf-8')).replace(', /', '').replace(', *,', ','), encoding='ascii')
	return pathFilename.with_stem('ascii')

def toCodon(pathFilename: Path) -> Path:
	if sys.platform == 'linux':
		buildCommand: list[str] = ['codon', 'build', '--exe', '--release', '--mcpu=native'
			, '--fast-math', '--enable-unsafe-fp-math', '--disable-exceptions'
			, '-o', str(pathFilename.with_suffix(''))
			, str(pathFilename)
		]

		subprocess.run(buildCommand, check=False)
		subprocess.run(['/usr/bin/strip', str(pathFilename.with_suffix(''))], check=False)

		sys.stdout.write(f"sudo systemd-run --unit={pathFilename.parent.name} --nice=-10 --property=CPUAffinity=0 {pathFilename.with_suffix('')}\n")

	return pathFilename.with_suffix('')

if __name__ == '__main__':
	toCodon(minify(assimilateEliminationCrease('module1')))
	toCodon(toASCII(assimilateEliminationCrease('module1')))
	pathFilename = packageSettings.pathPackage / '_e' / 'kitAST' / 'aa.py'
	toCodon(pathFilename)
