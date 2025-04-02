from collections.abc import Callable, Sequence
from mapFolding.someAssemblyRequired import ast_Identifier, astClassHasDOTvalue
from typing import Any
import ast

class Then:
	@staticmethod
	def allOf(listActions: Sequence[Callable[[ast.AST], Any]]) -> Callable[[ast.AST], ast.AST]:
		def workhorse(node: ast.AST) -> ast.AST:
			for action in listActions: action(node)
			return node
		return workhorse

	@staticmethod
	def appendTo(listOfAny: list[Any]) -> Callable[[ast.AST | ast_Identifier], list[Any]]:
		def workhorse(node: ast.AST | ast_Identifier) -> list[Any]:
			listOfAny.append(node)
			return listOfAny
		return workhorse

	@staticmethod
	def DOTarg(action: Callable[[Any], Any]) -> Callable[[ast.arg | ast.keyword], ast.arg | ast.keyword]:
		def workhorse(node: ast.arg | ast.keyword) -> ast.arg | ast.keyword:
			node.arg = action(node.arg)
			return node
		return workhorse
	@staticmethod
	def DOTfunc(action: Callable[[Any], Any]) -> Callable[[ast.Call], ast.Call]:
		def workhorse(node: ast.Call) -> ast.Call:
			node.func = action(node.func)
			return node
		return workhorse
	@staticmethod
	def DOTid(action: Callable[[Any], Any]) -> Callable[[ast.Name], ast.Name]:
		def workhorse(node: ast.Name) -> ast.Name:
			node.id = action(node.id)
			return node
		return workhorse
	@staticmethod
	def DOTtarget(action: Callable[[Any], Any]) -> Callable[[ast.AnnAssign | ast.AugAssign], ast.AnnAssign | ast.AugAssign]:
		def workhorse(node: ast.AnnAssign | ast.AugAssign) -> ast.AnnAssign | ast.AugAssign:
			node.target = action(node.target)
			return node
		return workhorse
	@staticmethod
	def DOTvalue(action: Callable[[Any], Any]) -> Callable[[astClassHasDOTvalue], astClassHasDOTvalue]:
		def workhorse(node: astClassHasDOTvalue) -> astClassHasDOTvalue:
			node.value = action(node.value)
			return node
		return workhorse

	@staticmethod
	def getIt(node: ast.AST) -> ast.AST | ast_Identifier:
		return node
	@staticmethod
	def insertThisAbove(list_astAST: Sequence[ast.AST]) -> Callable[[ast.AST], Sequence[ast.AST]]:
		return lambda aboveMe: [*list_astAST, aboveMe]
	@staticmethod
	def insertThisBelow(list_astAST: Sequence[ast.AST]) -> Callable[[ast.AST], Sequence[ast.AST]]:
		return lambda belowMe: [belowMe, *list_astAST]
	@staticmethod
	def removeIt(_node: ast.AST) -> None: return None
	@staticmethod
	def replaceWith(astAST: ast.AST | ast_Identifier) -> Callable[[ast.AST], ast.AST | ast_Identifier]:
		return lambda _replaceMe: astAST

	@staticmethod
	def updateKeyValueIn(key: Callable[..., Any], value: Callable[..., Any], dictionary: dict[Any, Any]) -> Callable[[ast.AST], dict[Any, Any]]:
		def workhorse(node: ast.AST) -> dict[Any, Any]:
			dictionary.setdefault(key(node), value(node))
			return dictionary
		return workhorse
