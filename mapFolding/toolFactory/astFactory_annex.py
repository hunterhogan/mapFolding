from typing import cast
import ast

grab_andDoAllOf: str ="""@staticmethod
def andDoAllOf(listOfActions: list[Callable[[NodeORattribute], NodeORattribute]]) -> Callable[[NodeORattribute], NodeORattribute]:
	def workhorse(node: NodeORattribute) -> NodeORattribute:
		for action in listOfActions:
			node = action(node)
		return node
	return workhorse
"""

grab_funcDOTidAttribute: str ="""@staticmethod
def funcDOTidAttribute(action: Callable[[ast_Identifier], Any]) -> Callable[[ImaCallToName], ImaCallToName]:
	def workhorse(node: ImaCallToName) -> ImaCallToName:
		node.func = grab.idAttribute(action)(node.func)
		return node
	return workhorse
"""

handmadeMethods_grab: list[ast.FunctionDef] = []
for string in [grab_andDoAllOf, grab_funcDOTidAttribute]:
	astModule = ast.parse(string)
	for node in ast.iter_child_nodes(astModule):
		if isinstance(node, ast.FunctionDef):
			handmadeMethods_grab.append(node)




"""Notes about the parse function:
Literal['exec'] = "exec"
def parse( source: str | ReadableBuffer, filename: str | ReadableBuffer | PathLike[Any] = "<unknown>", mode: Literal['exec'] = "exec", *, type_comments: bool = False, feature_version: int | tuple[int, int] | None = None, optimize: Literal[-1, 0, 1, 2] = -1
) -> Module: ...

mode: str = "exec",
def parse( source: str | ReadableBuffer, filename: str | ReadableBuffer | PathLike[Any] = "<unknown>", mode: str = "exec", *, type_comments: bool = False, feature_version: int | tuple[int, int] | None = None, optimize: Literal[-1, 0, 1, 2] = -1
) -> AST: ...

Creates an ast.expr type
mode: Literal['eval']
def parse( source: str | ReadableBuffer,
    filename: str | ReadableBuffer | PathLike[Any], mode: Literal['eval'], *, type_comments: bool = False, feature_version: int | tuple[int, int] | None = None, optimize: Literal[-1, 0, 1, 2] = -1
) -> Expression: ...

def parse( source: str | ReadableBuffer, *, mode: Literal['eval'], type_comments: bool = False, feature_version: int | tuple[int, int] | None = None, optimize: Literal[-1, 0, 1, 2] = -1
) -> Expression: ...

mode: Literal['single']
def parse( source: str | ReadableBuffer,
    filename: str | ReadableBuffer | PathLike[Any], mode: Literal['single'], *, type_comments: bool = False, feature_version: int | tuple[int, int] | None = None, optimize: Literal[-1, 0, 1, 2] = -1
) -> Interactive: ...

def parse( source: str | ReadableBuffer, *, mode: Literal['single'], type_comments: bool = False, feature_version: int | tuple[int, int] | None = None, optimize: Literal[-1, 0, 1, 2] = -1
) -> Interactive: ...

Python before 3.5
Literal['func_type']
def parse( source: str | ReadableBuffer, filename: str | ReadableBuffer | PathLike[Any], mode: Literal['func_type'], *, type_comments: bool = False, feature_version: int | tuple[int, int] | None = None, optimize: Literal[-1, 0, 1, 2] = -1 ) -> FunctionType: ...
def parse( source: str | ReadableBuffer, *, mode: Literal['func_type'], type_comments: bool = False, feature_version: int | tuple[int, int] | None = None, optimize: Literal[-1, 0, 1, 2] = -1 ) -> FunctionType: ...
"""
