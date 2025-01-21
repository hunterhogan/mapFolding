from mapFolding import indexMy, indexThe, indexTrack
import ast
import copy
import pathlib

def getDictionaryEnumValues():
    dictionaryEnumValues = {}
    for enumIndex in [indexMy, indexThe, indexTrack]:
        for memberName, memberValue in enumIndex._member_map_.items():
            dictionaryEnumValues[f"{enumIndex.__name__}.{memberName}.value"] = memberValue.value
    return dictionaryEnumValues

class RecursiveInlinerWithEnum(ast.NodeTransformer):
    def __init__(self, dictionaryFunctions, dictionaryEnumValues):
        self.dictionaryFunctions = dictionaryFunctions
        self.dictionaryEnumValues = dictionaryEnumValues
        self.processed = set()  # Track processed functions to avoid infinite recursion

    def inlineFunctionBody(self, functionName):
        if functionName in self.processed:
            return None

        self.processed.add(functionName)
        inlineDefinition = self.dictionaryFunctions[functionName]
        # Recursively process the function body
        for node in ast.walk(inlineDefinition):
            self.visit(node)
        return inlineDefinition

    def visit_Attribute(self, node):
        # Substitute enum identifiers (e.g., indexMy.leaf1ndex.value)
        if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name):
            enumPath = f"{node.value.value.id}.{node.value.attr}.{node.attr}"
            if enumPath in self.dictionaryEnumValues:
                return ast.Constant(value=self.dictionaryEnumValues[enumPath])
        return self.generic_visit(node)

    def visit_Call(self, node):
        callNode = self.generic_visit(node)
        if isinstance(callNode, ast.Call) and isinstance(callNode.func, ast.Name) and callNode.func.id in self.dictionaryFunctions:
            inlineDefinition = self.inlineFunctionBody(callNode.func.id)
            if inlineDefinition and inlineDefinition.body:
                lastStmt = inlineDefinition.body[-1]
                if isinstance(lastStmt, ast.Return) and lastStmt.value is not None:
                    return self.visit(lastStmt.value)
                elif isinstance(lastStmt, ast.Expr) and lastStmt.value is not None:
                    return self.visit(lastStmt.value)
                return None
        return callNode

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name) and node.value.func.id in self.dictionaryFunctions:
                inlineDefinition = self.inlineFunctionBody(node.value.func.id)
                if inlineDefinition:
                    return [self.visit(stmt) for stmt in inlineDefinition.body]
        return self.generic_visit(node)

def findRequiredImports(node):
    """Find all modules that need to be imported based on AST analysis.
    NOTE: due to hardcoding, this is a glorified regex. No, wait, this is less versatile than regex."""
    requiredImports = set()

    class ImportFinder(ast.NodeVisitor):
        def visit_Name(self, node):
            if node.id in {'numba'}:
                requiredImports.add(node.id)
            self.generic_visit(node)

        def visitDecorator(self, node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == 'jit':
                    requiredImports.add('numba')
            self.generic_visit(node)

    ImportFinder().visit(node)
    return requiredImports

def generateImports(requiredImports):
    """Generate import statements based on required modules."""
    importStatements = []

    # Map of module names to their import statements
    importMapping = {
        'numba': 'import numba',
    }

    for moduleName in sorted(requiredImports):
        if moduleName in importMapping:
            importStatements.append(importMapping[moduleName])

    return '\n'.join(importStatements)

def inlineFunctions(sourceCode, targetFunctionName, dictionaryEnumValues):
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

    # Generate imports
    requiredImports = findRequiredImports(nodeInlined)
    importStatements = generateImports(requiredImports)

    # Combine imports with inlined code
    inlinedCode = importStatements + '\n\n' + ast.unparse(ast.Module(body=[nodeInlined], type_ignores=[]))
    return inlinedCode

def Z0Z_inlineMapFolding():
    dictionaryEnumValues = getDictionaryEnumValues()

    pathFilenameSource = pathlib.Path("/apps/mapFolding/mapFolding/lovelace.py")
    codeSource = pathFilenameSource.read_text()

    listCallables = [
        'countInitialize',
        'countParallel',
        'countSequential',
    ]

    listPathFilenamesDestination: list[pathlib.Path] = []
    for callableTarget in listCallables:
        pathFilenameDestination = pathFilenameSource.with_stem(callableTarget)
        codeInlined = inlineFunctions(codeSource, callableTarget, dictionaryEnumValues)
        pathFilenameDestination.write_text(codeInlined)
        listPathFilenamesDestination.append(pathFilenameDestination)

    listNoNumba = [
        'countInitialize',
        'countSequential',
    ]

    listPathFilenamesNoNumba = []
    for pathFilename in listPathFilenamesDestination:
        if pathFilename.stem in listNoNumba:
            pathFilenameNoNumba = pathFilename.with_name(pathFilename.stem + 'NoNumba' + pathFilename.suffix)
        else:
            continue
        codeNoNumba = pathFilename.read_text()
        for codeLine in copy.copy(codeNoNumba.splitlines()):
            if 'numba' in codeLine:
                codeNoNumba = codeNoNumba.replace(codeLine, '')
        pathFilenameNoNumba.write_text(codeNoNumba)
        listPathFilenamesNoNumba.append(pathFilenameNoNumba)

if __name__ == '__main__':
    Z0Z_inlineMapFolding()
