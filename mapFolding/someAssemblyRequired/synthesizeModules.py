from mapFolding import indexMy, indexTrack, theDao, datatypeDefault, datatypeLarge, datatypeSmall
import ast
import pathlib
import inspect

algorithmSource = theDao

dictionaryDecorators={
    'countInitialize':
    f'@numba.jit((numba.{datatypeSmall}[:,:,::1], numba.{datatypeDefault}[::1], numba.{datatypeSmall}[::1], numba.{datatypeDefault}[:,::1]), parallel=False, boundscheck=False, cache=True, error_model="numpy", fastmath=True, looplift=False, nogil=True, nopython=True)\n',
    'countParallel':
    f'@numba.jit((numba.{datatypeSmall}[:,:,::1], numba.{datatypeLarge}[::1], numba.{datatypeDefault}[::1], numba.{datatypeSmall}[::1], numba.{datatypeDefault}[:,::1]), parallel=True, boundscheck=False, cache=True, error_model="numpy", fastmath=True, looplift=False, nogil=True, nopython=True)\n',
    'countSequential':
    f'@numba.jit((numba.{datatypeSmall}[:,:,::1], numba.{datatypeLarge}[::1], numba.{datatypeDefault}[::1], numba.{datatypeSmall}[::1], numba.{datatypeDefault}[:,::1]), parallel=False, boundscheck=False, cache=True, error_model="numpy", fastmath=True, looplift=False, nogil=True, nopython=True)\n',
    }

def getDictionaryEnumValues():
    dictionaryEnumValues = {}
    for enumIndex in [indexMy, indexTrack]:
        for memberName, memberValue in enumIndex._member_map_.items():
            dictionaryEnumValues[f"{enumIndex.__name__}.{memberName}.value"] = memberValue.value
    return dictionaryEnumValues

class RecursiveInlinerWithEnum(ast.NodeTransformer):
    """Process AST nodes to inline functions and substitute enum values.
    Also handles function decorators during inlining."""

    def __init__(self, dictionaryFunctions, dictionaryEnumValues):
        self.dictionaryFunctions = dictionaryFunctions
        self.dictionaryEnumValues = dictionaryEnumValues
        self.processed = set()

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
            if (inlineDefinition and inlineDefinition.body):
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
    importStatements = {'import numba', 'from mapFolding import indexMy, indexTrack'}

    importMapping = {
        'numba': 'import numba',
    }

    for moduleName in sorted(requiredImports):
        if moduleName in importMapping:
            importStatements.add(importMapping[moduleName])

    return '\n'.join(importStatements)

def inlineFunctions(sourceCode, targetFunctionName, dictionaryEnumValues, skipEnum=False):
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

    requiredImports = findRequiredImports(nodeInlined)
    importStatements = generateImports(requiredImports)

    lineNumbaDecorator = dictionaryDecorators[targetFunctionName]
    inlinedCode = importStatements + '\n\n' + lineNumbaDecorator + ast.unparse(ast.Module(body=[nodeInlined], type_ignores=[]))
    return inlinedCode

def unpackArrays(codeInlined: str, callableTarget: str) -> str:
    dictionaryReplaceScalars = {
        'my[indexMy.dimensionsTotal.value]': 'dimensionsTotal',
        'my[indexMy.dimensionsUnconstrained.value]': 'dimensionsUnconstrained',
        'my[indexMy.gap1ndex.value]': 'gap1ndex',
        'my[indexMy.gap1ndexCeiling.value]': 'gap1ndexCeiling',
        'my[indexMy.indexDimension.value]': 'indexDimension',
        # 'my[indexMy.indexLeaf.value]': 'indexLeaf',
        'my[indexMy.indexMiniGap.value]': 'indexMiniGap',
        'my[indexMy.leaf1ndex.value]': 'leaf1ndex',
        'my[indexMy.leafConnectee.value]': 'leafConnectee',
        # 'my[indexMy.taskDivisions.value]': 'taskDivisions',
        'my[indexMy.taskIndex.value]': 'taskIndex',
        # 'foldGroups[-1]': 'leavesTotal',
    }

    dictionaryReplaceArrays = {
        "track[indexTrack.leafAbove.value, ": 'leafAbove[',
        "track[indexTrack.leafBelow.value, ": 'leafBelow[',
        'track[indexTrack.countDimensionsGapped.value, ': 'countDimensionsGapped[',
        'track[indexTrack.gapRangeStart.value, ': 'gapRangeStart[',
    }

    ImaIndent = "    "
    linesInitialize = """"""

    for find, replace in dictionaryReplaceScalars.items():
        linesInitialize += f"{ImaIndent}{replace} = {find}\n"
        codeInlined = codeInlined.replace(find, replace)

    for find, replace in dictionaryReplaceArrays.items():
        linesInitialize += f"{ImaIndent}{replace[0:-1]} = {find[0:-2]}]\n"
        codeInlined = codeInlined.replace(find, replace)

    ourGuyOnTheInside = "    doFindGaps = True\n"
    linesInitialize = ourGuyOnTheInside + linesInitialize

    codeInlined = codeInlined.replace(ourGuyOnTheInside, linesInitialize)

    return codeInlined

def Z0Z_inlineMapFolding():
    dictionaryEnumValues = getDictionaryEnumValues()
    codeSource = inspect.getsource(algorithmSource)
    pathFilenameAlgorithm = pathlib.Path(inspect.getfile(algorithmSource))

    listCallables = [ 'countInitialize', 'countParallel', 'countSequential', ]

    listPathFilenamesDestination: list[pathlib.Path] = []
    for callableTarget in listCallables:
        skipEnum = (callableTarget == 'countInitialize')
        skipEnum = (callableTarget == 'countSequential')
        pathFilenameDestination = pathFilenameAlgorithm.parent / "someAssemblyRequired" / pathFilenameAlgorithm.with_stem(callableTarget).name
        codeInlined = inlineFunctions(codeSource, callableTarget, dictionaryEnumValues, skipEnum)
        codeUnpacked = unpackArrays(codeInlined, callableTarget)
        pathFilenameDestination.write_text(codeUnpacked)
        listPathFilenamesDestination.append(pathFilenameDestination)

if __name__ == '__main__':
    Z0Z_inlineMapFolding()
