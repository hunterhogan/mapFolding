from mapFolding import indexMy, indexTrack, getAlgorithmSource, ParametersNumba, parametersNumbaDEFAULT, hackSSOTdtype
from mapFolding import datatypeLargeDEFAULT, datatypeMediumDEFAULT, datatypeSmallDEFAULT, EnumIndices
import pathlib
import inspect
import numpy
import numba
from typing import Dict, Optional, List, Union, Sequence, Type, cast
import ast

algorithmSource = getAlgorithmSource()

class RecursiveInliner(ast.NodeTransformer):
    def __init__(self, dictionaryFunctions: Dict[str, ast.FunctionDef]):
        self.dictionaryFunctions = dictionaryFunctions
        self.processed = set()

    def inlineFunctionBody(self, functionName: str) -> Optional[ast.FunctionDef]:
        if (functionName in self.processed):
            return None

        self.processed.add(functionName)
        inlineDefinition = self.dictionaryFunctions[functionName]
        # Recursively process the function body
        for node in ast.walk(inlineDefinition):
            self.visit(node)
        return inlineDefinition

    def visit_Call(self, node: ast.Call) -> ast.AST:
        callNode = self.generic_visit(node)
        if (isinstance(callNode, ast.Call) and isinstance(callNode.func, ast.Name) and callNode.func.id in self.dictionaryFunctions):
            inlineDefinition = self.inlineFunctionBody(callNode.func.id)
            if (inlineDefinition and inlineDefinition.body):
                lastStmt = inlineDefinition.body[-1]
                if (isinstance(lastStmt, ast.Return) and lastStmt.value is not None):
                    return self.visit(lastStmt.value)
                elif (isinstance(lastStmt, ast.Expr) and lastStmt.value is not None):
                    return self.visit(lastStmt.value)
                return ast.Constant(value=None)
        return callNode

    def visit_Expr(self, node: ast.Expr) -> Union[ast.AST, List[ast.AST]]:
        if (isinstance(node.value, ast.Call)):
            if (isinstance(node.value.func, ast.Name) and node.value.func.id in self.dictionaryFunctions):
                inlineDefinition = self.inlineFunctionBody(node.value.func.id)
                if (inlineDefinition):
                    return [self.visit(stmt) for stmt in inlineDefinition.body]
        return self.generic_visit(node)

def decorateCallableWithNumba(astCallable: ast.FunctionDef, parallel: bool=False, **keywordArguments: Optional[str]) -> ast.FunctionDef:
    def makeNumbaParameterSignatureElement(signatureElement: ast.arg):
        if isinstance(signatureElement.annotation, ast.Subscript) and isinstance(signatureElement.annotation.slice, ast.Tuple):
            annotationShape = signatureElement.annotation.slice.elts[0]
            if isinstance(annotationShape, ast.Subscript) and isinstance(annotationShape.slice, ast.Tuple):
                shapeAsListSlices: Sequence[ast.expr] = [ast.Slice() for axis in range(len(annotationShape.slice.elts))]
                shapeAsListSlices[-1] = ast.Slice(step=ast.Constant(value=1))
                shapeAST = ast.Tuple(elts=list(shapeAsListSlices), ctx=ast.Load())
            else:
                shapeAST = ast.Slice(step=ast.Constant(value=1))

            annotationDtype = signatureElement.annotation.slice.elts[1]
            if (isinstance(annotationDtype, ast.Subscript) and isinstance(annotationDtype.slice, ast.Attribute)):
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

    listNumbaParameterSignature: Sequence[ast.expr] = []
    for parameter in astCallable.args.args:
        signatureElement = makeNumbaParameterSignatureElement(parameter)
        if (signatureElement):
            listNumbaParameterSignature.append(signatureElement)

    astArgsNumbaSignature = ast.Tuple(elts=listNumbaParameterSignature, ctx=ast.Load())

    if astCallable.name == 'countInitialize':
        parametersNumba = {}
    else:
        parametersNumba = parametersNumbaDEFAULT if not parallel else ParametersNumba({**parametersNumbaDEFAULT, 'parallel': True})
    listKeywordsNumbaSignature = [ast.keyword(arg=parameterName, value=ast.Constant(value=parameterValue)) for parameterName, parameterValue in parametersNumba.items()]

    astDecoratorNumba = ast.Call(func=ast.Attribute(value=ast.Name(id='numba', ctx=ast.Load()), attr='jit', ctx=ast.Load()), args=[astArgsNumbaSignature], keywords=listKeywordsNumbaSignature)

    astCallable.decorator_list = [astDecoratorNumba]
    return astCallable

class UnpackArrayAccesses(ast.NodeTransformer):
    """AST transformer that replaces array accesses with simpler variables."""

    def __init__(self, enumIndexClass: Type[EnumIndices], arrayName: str):
        self.enumIndexClass = enumIndexClass
        self.arrayName = arrayName
        self.substitutions = {}

    def extract_member_name(self, node: ast.AST) -> Optional[str]:
        """Recursively extract enum member name from any node in the AST."""
        if isinstance(node, ast.Attribute) and node.attr == 'value':
            innerAttribute = node.value
            while isinstance(innerAttribute, ast.Attribute):
                if (isinstance(innerAttribute.value, ast.Name) and innerAttribute.value.id == self.enumIndexClass.__name__):
                    return innerAttribute.attr
                innerAttribute = innerAttribute.value
        return None

    def transform_slice_element(self, node: ast.AST) -> ast.AST:
        """Transform any enum references within a slice element."""
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Attribute):
                member_name = self.extract_member_name(node.slice)
                if member_name:
                    return ast.Name(id=member_name, ctx=node.ctx)
            elif isinstance(node, ast.Tuple):
                # Handle tuple slices by transforming each element
                return ast.Tuple(elts=cast(List[ast.expr], [self.transform_slice_element(elt) for elt in node.elts]), ctx=node.ctx)
        elif isinstance(node, ast.Attribute):
            member_name = self.extract_member_name(node)
            if member_name:
                return ast.Name(id=member_name, ctx=ast.Load())
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        # Recursively visit any nested subscripts in value or slice
        node.value = self.visit(node.value)
        node.slice = self.visit(node.slice)
        # If node.value is not our arrayName, just return node
        if not (isinstance(node.value, ast.Name) and node.value.id == self.arrayName):
            return node

        # Handle scalar array access
        if isinstance(node.slice, ast.Attribute):
            memberName = self.extract_member_name(node.slice)
            if memberName:
                self.substitutions[memberName] = ('scalar', node)
                return ast.Name(id=memberName, ctx=ast.Load())

        # Handle array slice access
        if isinstance(node.slice, ast.Tuple) and node.slice.elts:
            firstElement = node.slice.elts[0]
            memberName = self.extract_member_name(firstElement)
            sliceRemainder = [self.visit(elem) for elem in node.slice.elts[1:]]
            if memberName:
                self.substitutions[memberName] = ('array', node)
                if len(sliceRemainder) == 0:
                    return ast.Name(id=memberName, ctx=ast.Load())
                return ast.Subscript(value=ast.Name(id=memberName, ctx=ast.Load()), slice=ast.Tuple(elts=sliceRemainder, ctx=ast.Load()) if len(sliceRemainder) > 1 else sliceRemainder[0], ctx=ast.Load())

        # If single-element tuple, unwrap
        if isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 1:
            node.slice = node.slice.elts[0]

        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node = cast(ast.FunctionDef, self.generic_visit(node))

        initializations = []
        for name, (kind, original_node) in self.substitutions.items():
            if kind == 'scalar':
                initializations.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=original_node))
            else:  # array
                initializations.append(
                    ast.Assign(
                        targets=[ast.Name(id=name, ctx=ast.Store())],
                        value=ast.Subscript(value=ast.Name(id=self.arrayName, ctx=ast.Load()),
                            slice=ast.Attribute(value=ast.Attribute(
                                    value=ast.Name(id=self.enumIndexClass.__name__, ctx=ast.Load()),
                                    attr=name, ctx=ast.Load()), attr='value', ctx=ast.Load()), ctx=ast.Load())))

        node.body = initializations + node.body
        return node

def inlineMapFoldingNumba(**keywordArguments: Optional[str]):
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

            if callableTarget == 'countSequential':
                myUnpacker = UnpackArrayAccesses(indexMy, 'my')
                callableDecorated = cast(ast.FunctionDef, myUnpacker.visit(callableDecorated))
                ast.fix_missing_locations(callableDecorated)

                trackUnpacker = UnpackArrayAccesses(indexTrack, 'track')
                callableDecorated = cast(ast.FunctionDef, trackUnpacker.visit(callableDecorated))
                ast.fix_missing_locations(callableDecorated)

            moduleAST = ast.Module(body=cast(List[ast.stmt], list(codeSourceImportStatements) + [callableDecorated]), type_ignores=[])
            ast.fix_missing_locations(moduleAST)
            moduleSource = ast.unparse(moduleAST)

            pathFilenameDestination = pathFilenameAlgorithm.parent / "syntheticModules" / pathFilenameAlgorithm.with_stem(callableTarget).name[5:None]
            pathFilenameDestination.write_text(moduleSource)
            listPathFilenamesDestination.append(pathFilenameDestination)

if __name__ == '__main__':
    inlineMapFoldingNumba()
