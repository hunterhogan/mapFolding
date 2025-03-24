from collections.abc import Callable, Sequence
from mapFolding.someAssemblyRequired import ast_Identifier, astClassHasDOTvalue, astMosDef, Make, nameDOTname
from typing import Any
import ast
"""
Semiotic notes:
In the `ast` package, some things that look and feel like a "name" are not `ast.Name` type. The following semiotics are a balance between technical precision and practical usage.

astName: always means `ast.Name`.
Name: uppercase, _should_ be interchangeable with astName, even in camelCase.
Hunter: ^^ did you do that ^^ ? Are you sure? You just fixed some "Name" identifiers that should have been "_name" because the wrong case confused you.
name: lowercase, never means `ast.Name`. In camelCase, I _should_ avoid using it in such a way that it could be confused with "Name", uppercase.
_Identifier: very strongly correlates with the private `ast._Identifier`, which is a `TypeAlias` for `str`.
identifier: lowercase, a general term that includes the above and other Python identifiers.
Identifier: uppercase, without the leading underscore should only appear in camelCase and means "identifier", lowercase.
namespace: lowercase, in dotted-names, such as `pathlib.Path` or `collections.abc`, "namespace" is the part before the dot.
Namespace: uppercase, should only appear in camelCase and means "namespace", lowercase.
"""

class Then:
	@staticmethod
	def allOf(listActions: Sequence[Callable[[ast.AST], Any]]) -> Callable[[ast.AST], ast.AST]:
		def workhorse(node: ast.AST) -> ast.AST:
			for action in listActions: action(node)
			return node
		return workhorse
	@staticmethod
	def append_target_idTo(list_Identifier: list[ast_Identifier]) -> Callable[[ast.AnnAssign], None]:
		return lambda node: list_Identifier.append(node.target.id) # type: ignore
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
