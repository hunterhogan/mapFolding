from __future__ import annotations  # ruff: ignore[undocumented-public-module]

from astToolkit import Be, Grab, NodeTourist, parsePathFilename2astModule, Then
from astToolkit.containers import makeDictionaryClassDef, makeDictionaryFunctionDef
from astToolkit.transformationTools import makeDictionaryMosDef
from collections import Counter
from humpy_cytoolz import concat
from mapFolding.kitAST import IfThis
from mapFolding.theSSOT import settingsPackage
from operator import methodcaller
from pprint import pprint
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterable
	from pathlib import Path
	import ast

#================== Settings and flow control ==================
scope: list[str] = [
	'core',
	'elimination',
	'2上nDimensional',
]

find: list[str] = [
	# 'astName',
	# 'CallName',
	# 'ClassDef',
	'FunctionDef',
	# 'MosDef',
]

containsString: list[str] = [
	'otal',  # total/Total
	'ount',  # count
	'umber',  # number
]

count: bool = False

#================== Scope =========================

boxOfPathFilenames: Iterable[Path] = []
if 'core' in scope:
	boxOfPathFilenames = concat([
		boxOfPathFilenames
		, settingsPackage.pathPackage.glob('*.py')
		, (settingsPackage.pathPackage / 'algorithms').rglob('*.py')
		, (settingsPackage.pathPackage / 'oeis').rglob('*.py')
		, (settingsPackage.pathPackage / 'kitAST').rglob('*.py')
	])

if 'elimination' in scope:
	boxOfPathFilenames = concat([
		boxOfPathFilenames
		, (settingsPackage.pathPackage / '_e').glob('*.py')
		, (settingsPackage.pathPackage / '_e.algorithms').rglob('*.py')
		, (settingsPackage.pathPackage / '_e.kitAST').rglob('*.py')
		, (settingsPackage.pathPackage / '_e.synthesized').rglob('*.py')
	])

if '2上nDimensional' in scope:
	boxOfPathFilenames = concat([
		boxOfPathFilenames
		, (settingsPackage.pathPackage / '_e._2上nDimensional').rglob('*.py')
	])

#================== Find =========================

boxOfIdentifiers: Iterable[str] = []
for pathFilename in boxOfPathFilenames:
	astModule: ast.Module = parsePathFilename2astModule(pathFilename)

	if 'astName' in find:
		NodeTourist(Be.Name, Grab.idAttribute(Then.appendTo(boxOfIdentifiers))).visit(astModule)
	if 'CallName' in find:
		NodeTourist(Be.Call.funcIs(Be.Name), Grab.funcAttribute(Grab.idAttribute(Then.appendTo(boxOfIdentifiers)))).visit(astModule)
	if 'ClassDef' in find:
		boxOfIdentifiers.extend(makeDictionaryClassDef(astModule))
	if 'FunctionDef' in find:
		boxOfIdentifiers.extend(makeDictionaryFunctionDef(astModule))
	if 'MosDef' in find:
		boxOfIdentifiers.extend(makeDictionaryMosDef(astModule))

#================== qq =========================
if count:
	idCounter: Counter[str] = Counter(boxOfIdentifiers)
	pprint(idCounter)  # ruff: ignore[p-print]
else:
	boxOfIdentifiers = map(methodcaller('removeprefix', '_'), boxOfIdentifiers)
	boxOfIdentifiers = sorted(set(boxOfIdentifiers))
	def has(identifier: str) -> bool:  # ruff: ignore[undocumented-public-function]
		return any(string in identifier for string in containsString)
	boxOfIdentifiers = filter(has, boxOfIdentifiers)
	pprint(list(boxOfIdentifiers), width=120)  # ruff: ignore[p-print]
