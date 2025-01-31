from mapFolding import indexMy, indexTrack, getAlgorithmSource, ParametersNumba, parametersNumbaDEFAULT, hackSSOTdtype
from mapFolding import datatypeLargeDEFAULT, datatypeMediumDEFAULT, datatypeSmallDEFAULT
import pathlib
import inspect
import numpy
import numba
from typing import Dict, Optional, List, Set, Union, Sequence
import ast

algorithmSource = getAlgorithmSource()

class RecursiveInliner(ast.NodeTransformer):
    def __init__(self, dictionaryFunctions: Dict[str, ast.FunctionDef]):
        self.dictionaryFunctions = dictionaryFunctions
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

def decorateCallableWithNumba(astCallable: ast.FunctionDef, parallel: bool=False, **keywordArguments: Optional[str]):
    def makeNumbaParameterSignatureElement(signatureElement: ast.arg):
        if isinstance(signatureElement.annotation, ast.Subscript) and isinstance(signatureElement.annotation.slice, ast.Tuple):

            annotationShape = signatureElement.annotation.slice.elts[0]
            if isinstance(annotationShape, ast.Subscript) and isinstance(annotationShape.slice, ast.Tuple):
                shapeAsListSlices = [ast.Slice() for axis in range(len(annotationShape.slice.elts))]
                shapeAsListSlices[-1] = ast.Slice(step=ast.Constant(value=1))
                shapeAST = ast.Tuple(elts=shapeAsListSlices, ctx=ast.Load())
            else:
                shapeAST = ast.Slice(step=ast.Constant(value=1))

            annotationDtype = signatureElement.annotation.slice.elts[1]
            if isinstance(annotationDtype, ast.Subscript) and isinstance(annotationDtype.slice, ast.Attribute):
                datatypeAST = annotationDtype.slice.attr
            else:
                datatypeAST = None

            ndarrayName = signatureElement.arg
            Z0Z_hackyStr = hackSSOTdtype[ndarrayName]
            Z0Z_hackyStr = Z0Z_hackyStr[0] + 'ata' + Z0Z_hackyStr[1:]
            datatype_attr = keywordArguments.get(Z0Z_hackyStr, None) or datatypeAST or eval(Z0Z_hackyStr+'DEFAULT')

            datatypeNumba = ast.Attribute(value=ast.Name(id='numba', ctx=ast.Load()), attr=datatype_attr, ctx=ast.Load())

            return ast.Subscript(value=datatypeNumba, slice=shapeAST, ctx=ast.Load())

    # callableSourceDecorators = [decorator for decorator in callableInlined.decorator_list]

    listNumbaParameterSignature: List[ast.Subscript] = []
    for parameter in astCallable.args.args:
        signatureElement = makeNumbaParameterSignatureElement(parameter)
        if signatureElement:
            listNumbaParameterSignature.append(signatureElement)

    astArgsNumbaSignature = ast.Tuple(elts=listNumbaParameterSignature, ctx=ast.Load())

    parametersNumba = parametersNumbaDEFAULT if not parallel else ParametersNumba({**parametersNumbaDEFAULT, 'parallel': True})
    listKeywordsNumbaSignature = [ast.keyword(arg=parameterName, value=ast.Constant(value=parameterValue)) for parameterName, parameterValue in parametersNumba.items()]

    astDecoratorNumba = ast.Call(func=ast.Attribute(value=ast.Name(id='numba', ctx=ast.Load()), attr='jit', ctx=ast.Load()), args=[astArgsNumbaSignature], keywords=listKeywordsNumbaSignature)

    astCallable.decorator_list = [astDecoratorNumba]
    return astCallable

def getDictionaryEnumValues() -> Dict[str, int]:
    dictionaryEnumValues = {}
    for enumIndex in [indexMy, indexTrack]:
        for memberName, memberValue in enumIndex._member_map_.items():
            dictionaryEnumValues[f"{enumIndex.__name__}.{memberName}.value"] = memberValue.value
    return dictionaryEnumValues

def unpackArrays(codeInlined: str) -> str:
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

def inlineMapFoldingNumba(**keywordArguments: Optional[str]):
    dictionaryEnumValues = getDictionaryEnumValues()
    codeSource = inspect.getsource(algorithmSource)
    pathFilenameAlgorithm = pathlib.Path(inspect.getfile(algorithmSource))

    listPathFilenamesDestination: list[pathlib.Path] = []
    listCallables = [ 'countInitialize', 'countParallel', 'countSequential', ]
    for callableTarget in listCallables:
        codeParsed: ast.Module = ast.parse(codeSource, type_comments=True)
        codeSourceImportStatements = {statement for statement in codeParsed.body if isinstance(statement, (ast.Import, ast.ImportFrom))}
        dictionaryFunctions = {statement.name: statement for statement in codeParsed.body if isinstance(statement, ast.FunctionDef)}
        callableInlinerWorkhorse = RecursiveInliner(dictionaryFunctions)
        parallel = callableTarget == 'countParallel'
        callableInlined = callableInlinerWorkhorse.inlineFunctionBody(callableTarget)
        if callableInlined:
            ast.fix_missing_locations(callableInlined)
            callableDecorated = decorateCallableWithNumba(callableInlined, parallel, **keywordArguments)

            importsRequired = "\n".join([ast.unparse(importStatement) for importStatement in codeSourceImportStatements])
            callableInlined = ast.unparse(callableDecorated)
            codeUnpacked = unpackArrays(callableInlined) if callableTarget == 'countSequential' else callableInlined
            # inlinedCode = ast.unparse(ast.Module(body=[nodeInlined], type_ignores=[]))

            pathFilenameDestination = pathFilenameAlgorithm.parent / "syntheticModules" / pathFilenameAlgorithm.with_stem(callableTarget).name
            pathFilenameDestination.write_text(importsRequired + "\n" + codeUnpacked)
            listPathFilenamesDestination.append(pathFilenameDestination)

if __name__ == '__main__':
    inlineMapFoldingNumba()
