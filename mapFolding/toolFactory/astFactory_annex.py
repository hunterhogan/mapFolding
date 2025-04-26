from typing import Any, cast, TypeAlias as typing_TypeAlias
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

astTypes_intORstr: str ="intORstr: typing_TypeAlias = Any"
astTypes_intORstrORtype_params: str ="intORstrORtype_params: typing_TypeAlias = Any"
astTypes_intORtype_params: str ="intORtype_params: typing_TypeAlias = Any"

handmadeTypeAlias_astTypes: list[ast.AnnAssign] = []
for string in [astTypes_intORstr, astTypes_intORstrORtype_params, astTypes_intORtype_params]:
	astModule = ast.parse(string)
	for node in ast.iter_child_nodes(astModule):
		if isinstance(node, ast.AnnAssign):
			handmadeTypeAlias_astTypes.append(node)


# print(ast.dump(ast.parse(ww)))
# from ast import *

"""
@staticmethod
def Import(moduleWithLogicalPath: str_nameDOTname, asname: ast_Identifier | None = None, **keywordArguments: int) -> ast.Import:
	return ast.Import(names=[Make.alias(moduleWithLogicalPath, asname)], **keywordArguments)

@staticmethod
def AnnAssign(target: ast.Attribute | ast.Name | ast.Subscript, annotation: ast.expr, value: ast.expr | None = None, **keywordArguments: int) -> ast.AnnAssign: # `simple: int`: uses a clever int-from-boolean to assign the correct value to the `simple` attribute. So, don't make it a method parameter.
	return ast.AnnAssign(target, annotation, value, simple=int(isinstance(target, ast.Name)), **keywordArguments)

"""
# @staticmethod
# def Attribute(value: ast.expr, *attribute: ast_Identifier, context: ast.expr_context = ast.Load(), **keywordArguments: int) -> ast.Attribute:
# 	""" If two `ast_Identifier` are joined by a dot `.`, they are _usually_ an `ast.Attribute`, but see `ast.ImportFrom`.
# 	Parameters:
# 		value: the part before the dot (e.g., `ast.Name`.)
# 		attribute: an `ast_Identifier` after a dot `.`; you can pass multiple `attribute` and they will be chained together.
# 	"""
# 	def addDOTattribute(chain: ast.expr, identifier: ast_Identifier, context: ast.expr_context, **keywordArguments: int) -> ast.Attribute:
# 		return ast.Attribute(value=chain, attr=identifier, ctx=context, **keywordArguments)
# 	buffaloBuffalo = addDOTattribute(value, attribute[0], context, **keywordArguments)
# 	for identifier in attribute[1:None]:
# 		buffaloBuffalo = addDOTattribute(buffaloBuffalo, identifier, context, **keywordArguments)
# 	return buffaloBuffalo


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
