# ruff: ignore[undocumented-public-module]
# DOCUMENT
from __future__ import annotations

from astToolkit import Be, Grab, NodeChanger, Then
from mapFolding.kitAST import IfThis
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import ast

def removeFunctionDef(identifier: str, node: ast.AST) -> None:  # ruff: ignore[undocumented-public-function]
	# DOCUMENT
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(identifier)), Then.removeIt).visit(node)

def renameFunctionDef(identifier: str, identifierNew: str, node: ast.AST) -> None:  # ruff: ignore[undocumented-public-function]
	# DOCUMENT
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(identifier)), Grab.nameAttribute(Then.replaceWith(identifierNew))).visit(node)

def renameName(identifier: str, identifierNew: str, node: ast.AST) -> None:  # ruff: ignore[undocumented-public-function]
	# DOCUMENT
	NodeChanger(Be.Name.idIs(IfThis.isIdentifier(identifier)), Grab.idAttribute(Then.replaceWith(identifierNew))).visit(node)
