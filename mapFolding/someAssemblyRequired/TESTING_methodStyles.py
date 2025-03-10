from collections.abc import Callable
import ast

class ifThis:
    class nameIsC:
        def __new__(cls, allegedly):
            return lambda node: isinstance(node, ast.Name) and node.id == allegedly

    @staticmethod
    def nameIsD(allegedly):
        return lambda node: (isinstance(node, ast.Name) and node.id == allegedly)

    @staticmethod
    def subscriptNameIsD(allegedly):
        return lambda node: (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == allegedly)

    @staticmethod
    def NameReallyIsD(allegedly):
        return ifThis.anyOfD(ifThis.nameIsD(allegedly), ifThis.subscriptNameIsD(allegedly))

    @staticmethod
    def anyOfD(*predicates):
        return lambda node: any(pred(node) for pred in predicates)
