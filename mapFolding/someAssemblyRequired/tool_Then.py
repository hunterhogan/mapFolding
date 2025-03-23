from collections.abc import Callable, Sequence
from mapFolding.someAssemblyRequired import (
	ast_Identifier,
	astClassHasDOTvalue,
	astMosDef,
	Make,
	nameDOTname,
)
from typing import Any
import ast

class Then:
	@staticmethod
	def Z0Z_actions(listActions: Sequence[Callable[[ast.AST], Any]]) -> Callable[[ast.AST], Sequence[Any]]:
		return lambda node: [action(node) for action in listActions]
	@staticmethod
	def append_targetTo(listName: list[ast.AST]) -> Callable[[ast.AnnAssign], None]:
		return lambda node: listName.append(node.target)
	@staticmethod
	def appendTo(listOfAny: list[Any]) -> Callable[[ast.AST], None]:
		return lambda node: listOfAny.append(node)
	@staticmethod
	def insertThisAbove(list_astAST: Sequence[ast.AST]) -> Callable[[ast.AST], Sequence[ast.AST]]:
		return lambda aboveMe: [*list_astAST, aboveMe]
	@staticmethod
	def insertThisBelow(list_astAST: Sequence[ast.AST]) -> Callable[[ast.AST], Sequence[ast.AST]]:
		return lambda belowMe: [belowMe, *list_astAST]
	@staticmethod
	def removeThis(_node: ast.AST) -> None: return None
	@staticmethod
	def replaceWith(astAST: ast.AST) -> Callable[[ast.AST], ast.AST]: return lambda _replaceMe: astAST
	@staticmethod
	def replaceDOTargWith(identifier: ast_Identifier) -> Callable[[ast.arg | ast.keyword], ast.arg | ast.keyword]:
		def workhorse(node: ast.arg | ast.keyword) -> ast.arg | ast.keyword:
			node.arg = identifier
			return node
		return workhorse
	@staticmethod
	def replaceDOTfuncWith(ast_expr: ast.expr) -> Callable[[ast.Call], ast.Call]:
		def workhorse(node: ast.Call) -> ast.Call:
			node.func = ast_expr
			return node
		return workhorse
	@staticmethod
	def replaceDOTidWith(identifier: ast_Identifier) -> Callable[[ast.Name], ast.Name]:
		def workhorse(node: ast.Name) -> ast.Name:
			node.id = identifier
			return node
		return workhorse
	@staticmethod
	def replaceDOTvalueWith(ast_expr: ast.expr) -> Callable[[astClassHasDOTvalue], astClassHasDOTvalue]:
		def workhorse(node: astClassHasDOTvalue) -> astClassHasDOTvalue:
			node.value = ast_expr
			return node
		return workhorse
	@staticmethod
	def updateThis(dictionaryOf_astMosDef: dict[ast_Identifier, astMosDef]) -> Callable[[astMosDef], astMosDef]:
		return lambda node: dictionaryOf_astMosDef.setdefault(node.name, node)
	from mapFolding.someAssemblyRequired.Z0Z_containers import LedgerOfImports
	@staticmethod
	def Z0Z_ledger(logicalPath: nameDOTname, ledger: LedgerOfImports) -> Callable[[ast.AnnAssign], None]:
		return lambda node: ledger.addImportFromAsStr(logicalPath, node.annotation.id) # type: ignore
	@staticmethod
	def Z0Z_appendKeywordMirroredTo(list_keyword: list[ast.keyword]) -> Callable[[ast.AnnAssign], None]:
		return lambda node: list_keyword.append(Make.ast_keyword(node.target.id, node.target)) # type: ignore
	@staticmethod
	def Z0Z_appendAnnAssignOf_nameDOTnameTo(identifier: ast_Identifier, list_nameDOTname: list[ast.AnnAssign]) -> Callable[[ast.AnnAssign], None]:
		return lambda node: list_nameDOTname.append(Make.astAnnAssign(node.target, node.annotation, Make.nameDOTname(identifier, node.target.id))) # type: ignore
