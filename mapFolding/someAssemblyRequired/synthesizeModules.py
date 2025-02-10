import ast
from typing import cast, Callable, List, Tuple, Dict, Optional

datatypeModuleScalar = 'numba'

# Example usage: Replace 'dimensionsTotal' with constant
# dimensions_replacer = NodeReplacer(
#     lambda n: isinstance(n, ast.Name) and n.id == "dimensionsTotal",
#     lambda _: ast.Constant(value=5)
# )

class ArgumentProcessor:
    """Unified argument processing using transformation rules"""
    def __init__(self, rules: List[Tuple[Callable[[ast.arg], bool], Callable]]):
        self.rules = rules  # (predicate, transformation)

    def process(self, func_def: ast.FunctionDef) -> ast.FunctionDef:
        for arg in func_def.args.args.copy():
            for predicate, transform in self.rules:
                if predicate(arg):
                    func_def = transform(func_def, arg)
        return func_def

# Example rule configuration
# array_arg_rule = (
#     lambda a: a.arg in ["track"],
#     lambda fn, arg: evaluate_array_access(fn, arg)
# )

# # Usage in transformers
# import_context = ImportFromTracker()
# import_context.addImportFrom("numba", "jit")

class AnnotationConverter(ast.NodeTransformer):
    """Convert type annotations to runtime initializations"""
    def __init__(self, import_tracker: ImportFromTracker):
        self.import_tracker = import_tracker

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        if isinstance(node.annotation, ast.Name):
            dtype = node.annotation.id
            self.import_tracker.addImportFrom(datatypeModuleScalar, dtype)
            return ast.Assign(
                targets=[node.target],
                value=ast.Call(
                    func=ast.Name(id=dtype),
                    args=[node.value] if node.value else [],
                    keywords=[]
                )
            )
        return node

class ArrayAccessUnpacker(ast.NodeTransformer):
    """Generic array index unpacking using enum mapping"""
    def __init__(self, index_mapping: Dict[str, int], array_name: str):
        self.index_mapping = index_mapping
        self.array_name = array_name
        self.substitutions = {}

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        node = cast(ast.Subscript, self.generic_visit(node))
        if isinstance(node.value, ast.Name) and node.value.id == self.array_name:
            if index := self._extract_index(node.slice):
                var_name = f"{self.array_name}_{index}"
                self.substitutions[var_name] = index
                return ast.Name(id=var_name)
        return node

    def _extract_index(self, slice_node: ast.AST) -> Optional[int]:
        # Index extraction logic using self.index_mapping
        pass
