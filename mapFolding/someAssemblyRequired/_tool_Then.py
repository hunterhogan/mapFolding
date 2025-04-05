from collections.abc import Callable, Sequence
from mapFolding.someAssemblyRequired import ast_Identifier, astClassHasDOTvalue
from typing import Any
import ast

class grab:
	"""
	class `grab`: bring the `Callable` to the node and its attribute or sub-node. Not for antecedents.
	class `DOT` : give only the attribute or sub-node to anything, including a `Callable`. Usable anywhere.
	"""
	@staticmethod
	def argAttribute(action: Callable[[ast_Identifier | None], ast_Identifier]) -> Callable[[ast.arg | ast.keyword], ast.arg | ast.keyword]:
		def workhorse(node: ast.arg | ast.keyword) -> ast.arg | ast.keyword:
			node.arg = action(node.arg)
			return node
		return workhorse

	@staticmethod
	def funcAttribute(action: Callable[[ast.expr], ast.expr]) -> Callable[[ast.Call], ast.Call]:
		def workhorse(node: ast.Call) -> ast.Call:
			node.func = action(node.func)
			return node
		return workhorse

	@staticmethod
	def idAttribute(action: Callable[[ast_Identifier], ast_Identifier]) -> Callable[[ast.Name], ast.Name]:
		def workhorse(node: ast.Name) -> ast.Name:
			node.id = action(node.id)
			return node
		return workhorse

	@staticmethod
	def valueAttribute(action: Callable[[Any | ast.expr | bool | None], Any]) -> Callable[[astClassHasDOTvalue], astClassHasDOTvalue]:
		def workhorse(node: astClassHasDOTvalue) -> astClassHasDOTvalue:
			node.value = action(node.value)
			return node
		return workhorse

class Then:
	@staticmethod
	def appendTo(listOfAny: list[Any]) -> Callable[[ast.AST | ast_Identifier], list[Any]]:
		def workhorse(node: ast.AST | ast_Identifier) -> list[Any]:
			listOfAny.append(node)
			return listOfAny
		return workhorse

	@staticmethod
	def extractIt(node: ast.AST) -> ast.AST | ast_Identifier:
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
	def replaceWith(astAST: Any) -> Callable[[Any], Any]:
		return lambda _replaceMe: astAST

	@staticmethod
	def updateKeyValueIn(key: Callable[..., Any], value: Callable[..., Any], dictionary: dict[Any, Any]) -> Callable[[ast.AST], dict[Any, Any]]:
		def workhorse(node: ast.AST) -> dict[Any, Any]:
			dictionary.setdefault(key(node), value(node))
			return dictionary
		return workhorse
