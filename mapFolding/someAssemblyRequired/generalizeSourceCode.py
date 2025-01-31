from mapFolding import datatypeLargeDEFAULT, datatypeMediumDEFAULT, datatypeSmallDEFAULT
from typing import Dict, Optional, List, Set, Union
import ast

class RecursiveInlinerWithEnum(ast.NodeTransformer):
    """Process AST nodes to inline functions and substitute enum values.
    Also handles function decorators during inlining."""

    def __init__(self, dictionaryFunctions: Dict[str, ast.FunctionDef], dictionaryEnumValues: Dict[str, int]) -> None:
        self.dictionaryFunctions = dictionaryFunctions
        self.dictionaryEnumValues = dictionaryEnumValues
        self.processed = set()

    def inlineFunctionBody(self, functionName: str) -> Optional[ast.FunctionDef]:
        if functionName in self.processed:
            return None

        self.processed.add(functionName)
        inlineDefinition = self.dictionaryFunctions[functionName]
        # Recursively process the function body
        for node in ast.walk(inlineDefinition):
            self.visit(node)
        return inlineDefinition

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        # Substitute enum identifiers (e.g., indexMy.leaf1ndex.value)
        if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name):
            enumPath = f"{node.value.value.id}.{node.value.attr}.{node.attr}"
            if enumPath in self.dictionaryEnumValues:
                return ast.Constant(value=self.dictionaryEnumValues[enumPath])
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        callNode = self.generic_visit(node)
        if isinstance(callNode, ast.Call) and isinstance(callNode.func, ast.Name) and callNode.func.id in self.dictionaryFunctions:
            inlineDefinition = self.inlineFunctionBody(callNode.func.id)
            if (inlineDefinition and inlineDefinition.body):
                lastStmt = inlineDefinition.body[-1]
                if isinstance(lastStmt, ast.Return) and lastStmt.value is not None:
                    return self.visit(lastStmt.value)
                elif isinstance(lastStmt, ast.Expr) and lastStmt.value is not None:
                    return self.visit(lastStmt.value)
                return ast.Constant(value=None)
        return callNode

    def visit_Expr(self, node: ast.Expr) -> Union[ast.AST, List[ast.AST]]:
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name) and node.value.func.id in self.dictionaryFunctions:
                inlineDefinition = self.inlineFunctionBody(node.value.func.id)
                if inlineDefinition:
                    return [self.visit(stmt) for stmt in inlineDefinition.body]
        return self.generic_visit(node)

def findRequiredImports(node: ast.AST) -> Set[str]:
    """Find all modules that need to be imported based on AST analysis.
    NOTE: due to hardcoding, this is a glorified regex. No, wait, this is less versatile than regex."""
    requiredImports = set()

    class ImportFinder(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if node.id in {'numba'}:
                requiredImports.add(node.id)
            self.generic_visit(node)

        def visitDecorator(self, node: ast.AST) -> None:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == 'jit':
                    requiredImports.add('numba')
            self.generic_visit(node)

    ImportFinder().visit(node)
    return requiredImports

def generateImports(requiredImports: Set[str]) -> str:
    """Generate import statements based on required modules."""
    importStatements = {'import numba', 'from mapFolding import indexMy, indexTrack'}

    importMapping = {
        'numba': 'import numba',
    }

    for moduleName in sorted(requiredImports):
        if moduleName in importMapping:
            importStatements.add(importMapping[moduleName])

    return '\n'.join(importStatements)

def makeInlineFunction(sourceCode: str, targetFunctionName: str, dictionaryEnumValues: Dict[str, int], skipEnum: bool=False, **keywordArguments: Optional[str]):
    datatypeLarge = keywordArguments.get('datatypeLarge', datatypeLargeDEFAULT)
    datatypeMedium = keywordArguments.get('datatypeMedium', datatypeMediumDEFAULT)
    datatypeSmall = keywordArguments.get('datatypeSmall', datatypeSmallDEFAULT)
    if skipEnum:
        dictionaryEnumValues = {}
    dictionaryParsed = ast.parse(sourceCode)
    dictionaryFunctions = {
        element.name: element
        for element in dictionaryParsed.body
        if isinstance(element, ast.FunctionDef)
    }
    nodeTarget = dictionaryFunctions[targetFunctionName]
    nodeInliner = RecursiveInlinerWithEnum(dictionaryFunctions, dictionaryEnumValues)
    nodeInlined = nodeInliner.visit(nodeTarget)
    ast.fix_missing_locations(nodeInlined)
    callableInlinedDecorators = [decorator for decorator in nodeInlined.decorator_list]

    requiredImports = findRequiredImports(nodeInlined)
    importStatements = generateImports(requiredImports)
    importsRequired = importStatements
    dictionaryDecoratorsNumba={
        'countInitialize':
        f'@numba.jit((numba.{datatypeSmall}[:,:,::1], numba.{datatypeMedium}[::1], numba.{datatypeSmall}[::1], numba.{datatypeMedium}[:,::1]), parallel=False, boundscheck=False, cache=True, error_model="numpy", fastmath=True, looplift=False, nogil=True, nopython=True)\n',
        'countParallel':
        f'@numba.jit((numba.{datatypeSmall}[:,:,::1], numba.{datatypeLarge}[::1], numba.{datatypeMedium}[::1], numba.{datatypeSmall}[::1], numba.{datatypeMedium}[:,::1]), parallel=True, boundscheck=False, cache=True, error_model="numpy", fastmath=True, looplift=False, nogil=True, nopython=True)\n',
        'countSequential':
        f'@numba.jit((numba.{datatypeSmall}[:,:,::1], numba.{datatypeLarge}[::1], numba.{datatypeMedium}[::1], numba.{datatypeSmall}[::1], numba.{datatypeMedium}[:,::1]), parallel=False, boundscheck=False, cache=True, error_model="numpy", fastmath=True, looplift=False, nogil=True, nopython=True)\n',
        }

    lineNumbaDecorator = dictionaryDecoratorsNumba[targetFunctionName]

    # inlinedCode = ast.unparse(ast.Module(body=[nodeInlined], type_ignores=[]))
    callableInlined = lineNumbaDecorator + ast.unparse(nodeInlined)
    return (callableInlined, callableInlinedDecorators, importsRequired)
