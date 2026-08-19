from __future__ import annotations

from astToolkit import Be, Grab, NodeChanger, Then
from mapFolding.kitAST import IfThis
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import ast

def removeFunctionDef(identifier: str, node: ast.AST) -> None:
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(identifier)), Then.removeIt).visit(node)

def renameFunctionDef(identifier: str, identifierNew: str, node: ast.AST) -> None:
	NodeChanger(Be.FunctionDef.nameIs(IfThis.isIdentifier(identifier)), Grab.nameAttribute(Then.replaceWith(identifierNew))).visit(node)
