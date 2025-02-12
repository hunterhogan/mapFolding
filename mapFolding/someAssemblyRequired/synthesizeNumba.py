from mapFolding import (
    computationState,
    EnumIndices,
    getAlgorithmSource,
    getFilenameFoldsTotal,
    getPathFilenameFoldsTotal,
    getPathJobRootDEFAULT,
    getPathSyntheticModules,
    hackSSOTdatatype,
    indexMy,
    indexTrack,
    moduleOfSyntheticModules,
    myPackageNameIs,
    ParametersNumba,
    parametersNumbaDEFAULT,
    parametersNumbaFailEarly,
    parametersNumbaSuperJit,
    parametersNumbaSuperJitParallel,
    setDatatypeElephino,
    setDatatypeFoldsTotal,
    setDatatypeLeavesTotal,
    setDatatypeModule,
)
from mapFolding.someAssemblyRequired import makeStateJob
from types import ModuleType
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Set, Tuple, Type, Union
import ast
import autoflake
import collections
import inspect
import more_itertools
import numba
import numpy
import os
import pathlib
import python_minifier

youOughtaKnow = collections.namedtuple('youOughtaKnow', ['callableSynthesized', 'pathFilenameForMe', 'astForCompetentProgrammers'])

class UniversalImportTracker:
    def __init__(self):
        self.dictionaryImportFrom = collections.defaultdict(set)
        self.setImport = set()

    def addAst(self, astImport_: Union[ast.Import, ast.ImportFrom]) -> None:
        if isinstance(astImport_, ast.Import):
            for alias in astImport_.names:
                self.setImport.add(alias.name)
        elif isinstance(astImport_, ast.ImportFrom):
            self.dictionaryImportFrom[astImport_.module].update(alias.name for alias in astImport_.names)

    def addImportFromStr(self, module: str, name: str) -> None:
        self.dictionaryImportFrom[module].add(name)

    def addImportStr(self, name: str) -> None:
        self.setImport.add(name)

    def makeListAst(self) -> List[Union[ast.ImportFrom, ast.Import]]:
        listAstImportFrom = [ast.ImportFrom(module=module, names=[ast.alias(name=name, asname=None)], level=0) for module, names in self.dictionaryImportFrom.items() for name in names]
        listAstImport = [ast.Import(names=[ast.alias(name=name, asname=None)]) for name in self.setImport]
        return listAstImportFrom + listAstImport

class NodeReplacer(ast.NodeTransformer):
    """Base class for configurable node replacement"""
    def __init__(self, findMe, nodeReplacementBuilder):
        self.findMe = findMe
        self.nodeReplacementBuilder = nodeReplacementBuilder

    def visit(self, node: ast.AST) -> ast.AST:
        if self.findMe(node):
            return self.nodeReplacementBuilder(node)
        return super().visit(node)

class ArgumentProcessor:
    """Unified argument processing using transformation rules"""
    def __init__(self, rules: List[Tuple[Callable[[ast.arg], bool], Callable]]):
        self.rules = rules  # (predicate, transformation)

    def process(self, FunctionDef: ast.FunctionDef) -> ast.FunctionDef:
        for arg in FunctionDef.args.args.copy():
            for predicate, transform in self.rules:
                if predicate(arg):
                    FunctionDef = transform(FunctionDef, arg)
        return FunctionDef

class RecursiveInliner(ast.NodeTransformer):
    """
    Class RecursiveInliner:
        A custom AST NodeTransformer designed to recursively inline function calls from a given dictionary
        of function definitions into the AST. Once a particular function has been inlined, it is marked
        as completed to avoid repeated inlining. This transformation modifies the AST in-place by substituting
        eligible function calls with the body of their corresponding function.
        Attributes:
            dictionaryFunctions (Dict[str, ast.FunctionDef]):
                A mapping of function name to its AST definition, used as a source for inlining.
            callablesCompleted (Set[str]):
                A set to track function names that have already been inlined to prevent multiple expansions.
        Methods:
            inlineFunctionBody(callableTargetName: str) -> Optional[ast.FunctionDef]:
                Retrieves the AST definition for a given function name from dictionaryFunctions
                and recursively inlines any function calls within it. Returns the function definition
                that was inlined or None if the function was already processed.
            visit_Call(callNode: ast.Call) -> ast.AST:
                Inspects calls within the AST. If a function call matches one in dictionaryFunctions,
                it is replaced by the inlined body. If the last statement in the inlined body is a return
                or an expression, that value or expression is substituted; otherwise, a constant is returned.
            visit_Expr(node: ast.Expr) -> Union[ast.AST, List[ast.AST]]:
                Handles expression nodes in the AST. If the expression is a function call from
                dictionaryFunctions, its statements are expanded in place, effectively inlining
                the called function's statements into the surrounding context.
    """
    def __init__(self, dictionaryFunctions: Dict[str, ast.FunctionDef]):
        self.dictionaryFunctions = dictionaryFunctions
        self.callablesCompleted: Set[str] = set()

    def inlineFunctionBody(self, callableTargetName: str) -> Optional[ast.FunctionDef]:
        if (callableTargetName in self.callablesCompleted):
            return None

        self.callablesCompleted.add(callableTargetName)
        inlineDefinition = self.dictionaryFunctions[callableTargetName]
        for astNode in ast.walk(inlineDefinition):
            self.visit(astNode)
        return inlineDefinition

    def visit_Call(self, node: ast.Call) -> ast.AST:
        callNodeVisited = self.generic_visit(node)
        if (isinstance(callNodeVisited, ast.Call) and isinstance(callNodeVisited.func, ast.Name) and callNodeVisited.func.id in self.dictionaryFunctions):
            inlineDefinition = self.inlineFunctionBody(callNodeVisited.func.id)
            if (inlineDefinition and inlineDefinition.body):
                statementTerminating = inlineDefinition.body[-1]
                if (isinstance(statementTerminating, ast.Return) and statementTerminating.value is not None):
                    return self.visit(statementTerminating.value)
                elif (isinstance(statementTerminating, ast.Expr) and statementTerminating.value is not None):
                    return self.visit(statementTerminating.value)
                return ast.Constant(value=None)
        return callNodeVisited

    def visit_Expr(self, node: ast.Expr) -> Union[ast.AST, List[ast.AST]]:
        if (isinstance(node.value, ast.Call)):
            if (isinstance(node.value.func, ast.Name) and node.value.func.id in self.dictionaryFunctions):
                inlineDefinition = self.inlineFunctionBody(node.value.func.id)
                if (inlineDefinition):
                    return [self.visit(stmt) for stmt in inlineDefinition.body]
        return self.generic_visit(node)

class UnpackArrayAccesses(ast.NodeTransformer):
    """
    A class that transforms array accesses using enum indices into local variables.

    This AST transformer identifies array accesses using enum indices and replaces them
    with local variables, adding initialization statements at the start of functions.

    Parameters:
        enumIndexClass (Type[EnumIndices]): The enum class used for array indexing
        arrayName (str): The name of the array being accessed

    Attributes:
        enumIndexClass (Type[EnumIndices]): Stored enum class for index lookups
        arrayName (str): Name of the array being transformed
        substitutions (dict): Tracks variable substitutions and their original nodes

    The transformer handles two main cases:
    1. Scalar array access - array[EnumIndices.MEMBER]
    2. Array slice access - array[EnumIndices.MEMBER, other_indices...]
    For each identified access pattern, it:
    1. Creates a local variable named after the enum member
    2. Adds initialization code at function start
    3. Replaces original array access with the local variable
    """

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

def Z0Z_UnhandledDecorators(astCallable: ast.FunctionDef) -> ast.FunctionDef:
    # TODO: more explicit handling of decorators. I'm able to ignore this because I know `algorithmSource` doesn't have any decorators.
    for decoratorItem in astCallable.decorator_list.copy():
        import warnings
        astCallable.decorator_list.remove(decoratorItem)
        warnings.warn(f"Removed decorator {ast.unparse(decoratorItem)} from {astCallable.name}")
    return astCallable
def isThisNodeNumbaJitCall(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == decoratorCallable)
def isThisNodeJitCall(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == decoratorCallable)
def isThisNodeNumbaJitDecorator(node: ast.AST) -> bool:
    return isThisNodeNumbaJitCall(node) or isThisNodeJitCall(node)
def Z0Z_generalizeThis(FunctionDefTarget: ast.FunctionDef, parametersNumba: Optional[ParametersNumba]=None) -> Tuple[ast.FunctionDef, ParametersNumba | None]:
    def recycleParametersNumba(decorator: ast.Call) -> Dict[str, Any]:
        parametersNumbaExtracted: Dict[str, Any] = {}
        for keywordItem in decorator.keywords:
            if isinstance(keywordItem.value, ast.Constant) and keywordItem.arg is not None:
                parametersNumbaExtracted[keywordItem.arg] = keywordItem.value.value
        return parametersNumbaExtracted

    for decorator in FunctionDefTarget.decorator_list.copy():
        if isThisNodeNumbaJitDecorator(decorator):
            decorator = cast(ast.Call, decorator)
            if parametersNumba is None:
                parametersNumbaSherpa = recycleParametersNumba(decorator)
                if (HunterIsSureThereAreBetterWaysToDoThis := True):
                    if parametersNumbaSherpa:
                        parametersNumba = cast(ParametersNumba, parametersNumbaSherpa)
        FunctionDefTarget.decorator_list.remove(decorator)

    return FunctionDefTarget, parametersNumba

def decorateCallableWithNumba(FunctionDefTarget: ast.FunctionDef, allImports: UniversalImportTracker, parametersNumba: Optional[ParametersNumba]=None) -> Tuple[ast.FunctionDef, UniversalImportTracker]:
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
            Z0Z_hacky_dtype = hackSSOTdatatype(ndarrayName)
            datatype_attr = datatypeAST or Z0Z_hacky_dtype
            allImports.addImportFromStr(datatypeModuleDecorator, datatype_attr)
            datatypeNumba = ast.Name(id=datatype_attr, ctx=ast.Load())

            return ast.Subscript(value=datatypeNumba, slice=shapeAST, ctx=ast.Load())

    FunctionDefTarget, parametersNumba = Z0Z_generalizeThis(FunctionDefTarget, parametersNumba)

    FunctionDefTarget = Z0Z_UnhandledDecorators(FunctionDefTarget)

    listDecorator_args: Sequence[ast.expr] = []

    listForNumba_arg_named_signature_or_function: Sequence[ast.expr] = []
    for parameter in FunctionDefTarget.args.args:
        signatureElement = makeNumbaParameterSignatureElement(parameter)
        if signatureElement:
            listForNumba_arg_named_signature_or_function.append(signatureElement)

    if FunctionDefTarget.returns and isinstance(FunctionDefTarget.returns, ast.Name):
        theReturn: ast.Name = FunctionDefTarget.returns
        listDecorator_args = [cast(ast.expr,
                                ast.Call(
                                    func=ast.Name(id=theReturn.id, ctx=ast.Load())
                                    , args=listForNumba_arg_named_signature_or_function if listForNumba_arg_named_signature_or_function else []
                                    , keywords=[]
                                )
                            )]

    elif listForNumba_arg_named_signature_or_function:
        listDecorator_args = [cast(ast.expr, ast.Tuple(elts=listForNumba_arg_named_signature_or_function, ctx=ast.Load()))]

    if parametersNumba is None:
        parametersNumba = parametersNumbaDEFAULT
    listDecoratorKeywords = [ast.keyword(arg=parameterName, value=ast.Constant(value=parameterValue)) for parameterName, parameterValue in parametersNumba.items()]

    allImports.addImportFromStr(decoratorModule, decoratorCallable)
    astDecorator = ast.Call(
        func=ast.Name(id=decoratorCallable, ctx=ast.Load())
        , args=listDecorator_args if listDecorator_args else []
        , keywords=listDecoratorKeywords)

    FunctionDefTarget.decorator_list = [astDecorator]
    return FunctionDefTarget, allImports

def makeDecoratorJobNumba(FunctionDefTarget: ast.FunctionDef, allImports: UniversalImportTracker, parametersNumba: Optional[ParametersNumba]=None) -> Tuple[ast.FunctionDef, UniversalImportTracker]:

    def convertToPlainJit(node: ast.Call) -> ast.Call:
        node.func = ast.Name(id=decoratorCallable, ctx=ast.Load())
        return node

    FunctionDefTarget, parametersNumba = Z0Z_generalizeThis(FunctionDefTarget, parametersNumba)

    FunctionDefTarget, allImports = decorateCallableWithNumba(FunctionDefTarget, allImports, parametersNumba)
    if isThisNodeNumbaJitCall(FunctionDefTarget.decorator_list[0]):
        FunctionDefTarget.decorator_list[0] = convertToPlainJit(cast(ast.Call, FunctionDefTarget.decorator_list[0]))

    return FunctionDefTarget, allImports

def inlineOneCallable(pythonSource: str, callableTarget: str) -> str | None:
    astModule: ast.Module = ast.parse(pythonSource, type_comments=True)
    allImports = UniversalImportTracker()

    for statement in astModule.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            allImports.addAst(statement)

    dictionaryFunctionDef = {statement.name: statement for statement in astModule.body if isinstance(statement, ast.FunctionDef)}
    callableInlinerWorkhorse = RecursiveInliner(dictionaryFunctionDef)
    FunctionDefTarget = callableInlinerWorkhorse.inlineFunctionBody(callableTarget)

    if FunctionDefTarget:
        ast.fix_missing_locations(FunctionDefTarget)
        parametersNumba = None
        match callableTarget:
            case 'countParallel':
                parametersNumba = parametersNumbaSuperJitParallel
            case 'countSequential':
                parametersNumba = parametersNumbaSuperJit
            case 'countInitialize':
                parametersNumba = parametersNumbaDEFAULT

        FunctionDefTarget, allImports = decorateCallableWithNumba(FunctionDefTarget, allImports, parametersNumba)

        if callableTarget == 'countSequential':
            unpackerMy = UnpackArrayAccesses(indexMy, 'my')
            FunctionDefTarget = cast(ast.FunctionDef, unpackerMy.visit(FunctionDefTarget))
            ast.fix_missing_locations(FunctionDefTarget)

            unpackerTrack = UnpackArrayAccesses(indexTrack, 'track')
            FunctionDefTarget = cast(ast.FunctionDef, unpackerTrack.visit(FunctionDefTarget))
            ast.fix_missing_locations(FunctionDefTarget)

        moduleAST = ast.Module(body=cast(List[ast.stmt], allImports.makeListAst() + [FunctionDefTarget]), type_ignores=[])
        ast.fix_missing_locations(moduleAST)
        moduleSource = ast.unparse(moduleAST)
        return moduleSource

def makeDispatcherNumba(pythonSource: str, callableTarget: str, listStuffYouOughtaKnow: List[youOughtaKnow]) -> str:
    astSource = ast.parse(pythonSource)
    allImports = UniversalImportTracker()

    for statement in astSource.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            allImports.addAst(statement)

    for stuff in listStuffYouOughtaKnow:
        statement = stuff.astForCompetentProgrammers
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            allImports.addAst(statement)

    FunctionDefTarget = next((node for node in astSource.body if isinstance(node, ast.FunctionDef) and node.name == callableTarget), None)

    if not FunctionDefTarget:
        raise ValueError(f"Could not find function {callableTarget} in source code")

    FunctionDefTarget, allImports = decorateCallableWithNumba(FunctionDefTarget, allImports, parametersNumbaFailEarly)

    astModule = ast.Module( body=cast(List[ast.stmt], allImports.makeListAst()
                + [FunctionDefTarget]), type_ignores=[])

    ast.fix_missing_locations(astModule)
    return ast.unparse(astModule)

def makeNumbaOptimizedFlow(listCallablesInline: List[str], callableDispatcher: Optional[str] = None, algorithmSource: Optional[ModuleType] = None) -> None:
    if not algorithmSource:
        algorithmSource = getAlgorithmSource()

    formatModuleNameDEFAULT = "numba_{callableTarget}"

    # When I am a more competent programmer, I will make getPathFilenameWrite dependent on makeAstImport or vice versa,
    # so the name of the physical file doesn't get out of whack with the name of the logical module.
    def getPathFilenameWrite(callableTarget: str
                            , pathWrite: Optional[pathlib.Path] = None
                            , formatFilenameWrite: Optional[str] = None
                            ) -> pathlib.Path:
        if not pathWrite:
            pathWrite = getPathSyntheticModules()
        if not formatFilenameWrite:
            formatFilenameWrite = formatModuleNameDEFAULT + '.py'

        pathFilename = pathWrite  / formatFilenameWrite.format(callableTarget=callableTarget)
        return pathFilename

    def makeAstImport(callableTarget: str
                        , packageName: Optional[str] = None
                        , subPackageName: Optional[str] = None
                        , moduleName: Optional[str] = None
                        , astNodeLogicalPathThingy: Optional[ast.AST] = None
                        ) -> ast.ImportFrom:
        """Creates import AST node for synthetic modules."""
        if astNodeLogicalPathThingy is None:
            if packageName is None:
                packageName = myPackageNameIs
            if subPackageName is None:
                subPackageName = moduleOfSyntheticModules
            if moduleName is None:
                moduleName = formatModuleNameDEFAULT.format(callableTarget=callableTarget)
            module=f'{packageName}.{subPackageName}.{moduleName}'
        else:
            module = str(astNodeLogicalPathThingy)
        return ast.ImportFrom(
            module=module,
            names=[ast.alias(name=callableTarget, asname=None)],
            level=0
        )

    listStuffYouOughtaKnow: List[youOughtaKnow] = []

    for callableTarget in listCallablesInline:
        pythonSource = inspect.getsource(algorithmSource)
        pythonSource = inlineOneCallable(pythonSource, callableTarget)
        if not pythonSource:
            raise Exception("Pylance, OMG! The sky is falling!")

        pathFilename = getPathFilenameWrite(callableTarget)

        listStuffYouOughtaKnow.append(youOughtaKnow(
            callableSynthesized=callableTarget,
            pathFilenameForMe=pathFilename,
            astForCompetentProgrammers=makeAstImport(callableTarget)
        ))
        pythonSource = autoflake.fix_code(pythonSource, ['mapFolding', 'numba', 'numpy'])
        pathFilename.write_text(pythonSource)

    # Generate dispatcher if requested
    if callableDispatcher:
        pythonSource = inspect.getsource(algorithmSource)
        pythonSource = makeDispatcherNumba(pythonSource, callableDispatcher, listStuffYouOughtaKnow)
        if not pythonSource:
            raise Exception("Pylance, OMG! The sky is falling!")

        pathFilename = getPathFilenameWrite(callableDispatcher)

        listStuffYouOughtaKnow.append(youOughtaKnow(
            callableSynthesized=callableDispatcher,
            pathFilenameForMe=pathFilename,
            astForCompetentProgrammers=makeAstImport(callableDispatcher)
        ))
        pythonSource = autoflake.fix_code(pythonSource, ['mapFolding', 'numba', 'numpy'])
        pathFilename.write_text(pythonSource)

def makeStrRLEcompacted(arrayTarget: numpy.ndarray, identifierName: Optional[str]=None) -> str:
    """Converts a NumPy array into a compressed string representation using run-length encoding (RLE).

    This function takes a NumPy array and converts it into an optimized string representation by:
    1. Compressing consecutive sequences of numbers into range objects
    2. Minimizing repeated zeros using array multiplication syntax
    3. Converting the result into a valid Python array initialization statement

    Parameters:
        arrayTarget (numpy.ndarray): The input NumPy array to be converted
        identifierName (str): The variable name to use in the output string

    Returns:
        str: A string containing Python code that recreates the input array in compressed form.
            Format: "{identifierName} = numpy.array({compressed_data}, dtype=numpy.{dtype})"
    """

    def compressRangesNDArrayNoFlatten(arraySlice):
        if isinstance(arraySlice, numpy.ndarray) and arraySlice.ndim > 1:
            return [compressRangesNDArrayNoFlatten(arraySlice[index]) for index in range(arraySlice.shape[0])]
        elif isinstance(arraySlice, numpy.ndarray) and arraySlice.ndim == 1:
            listWithRanges = []
            for group in more_itertools.consecutive_groups(arraySlice.tolist()):
                ImaSerious = list(group)
                if len(ImaSerious) <= 4:
                    listWithRanges += ImaSerious
                else:
                    ImaRange = [range(ImaSerious[0], ImaSerious[-1] + 1)]
                    listWithRanges += ImaRange
            return listWithRanges
        return arraySlice

    arrayAsNestedLists = compressRangesNDArrayNoFlatten(arrayTarget)

    stringMinimized = python_minifier.minify(str(arrayAsNestedLists))
    commaZeroMaximum = arrayTarget.shape[-1] - 1
    stringMinimized = stringMinimized.replace('[0' + ',0'*commaZeroMaximum + ']', '[0]*'+str(commaZeroMaximum+1))
    for countZeros in range(commaZeroMaximum, 2, -1):
        stringMinimized = stringMinimized.replace(',0'*countZeros + ']', ']+[0]*'+str(countZeros))

    stringMinimized = stringMinimized.replace('range', '*range')

    if identifierName:
        return f"{identifierName} = array({stringMinimized}, dtype={arrayTarget.dtype})"
    return stringMinimized

def moveArrayTo_body(FunctionDefTarget: ast.FunctionDef, astArg: ast.arg, argData: numpy.ndarray) -> ast.FunctionDef:
    arrayType = type(argData)
    moduleConstructor = arrayType.__module__
    constructorName = arrayType.__name__
    # NOTE hack
    constructorName = constructorName.replace('ndarray', 'array')
    argData_dtype: numpy.dtype = argData.dtype
    argData_dtypeName = argData.dtype.name

    allImports.addImportFromStr(moduleConstructor, constructorName)
    allImports.addImportFromStr(moduleConstructor, argData_dtypeName)

    onlyDataRLE = makeStrRLEcompacted(argData)
    astStatement = cast(ast.Expr, ast.parse(onlyDataRLE).body[0])
    dataAst = astStatement.value

    arrayCall = ast.Call(
        func=ast.Name(id=constructorName, ctx=ast.Load())
        , args=[dataAst]
        , keywords=[ast.keyword(arg='dtype' , value=ast.Name(id=argData_dtypeName , ctx=ast.Load()) ) ] )

    assignment = ast.Assign( targets=[ast.Name(id=astArg.arg, ctx=ast.Store())], value=arrayCall )
    FunctionDefTarget.body.insert(0, assignment)
    FunctionDefTarget.args.args.remove(astArg)

    return FunctionDefTarget

def evaluateArrayIn_body(FunctionDefTarget: ast.FunctionDef, astArg: ast.arg, argData: numpy.ndarray) -> ast.FunctionDef:
    arrayType = type(argData)
    moduleConstructor = arrayType.__module__
    constructorName = arrayType.__name__
    # NOTE hack
    constructorName = constructorName.replace('ndarray', 'array')
    allImports.addImportFromStr(moduleConstructor, constructorName)

    for stmt in FunctionDefTarget.body.copy():
        if isinstance(stmt, ast.Assign):
            if isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Subscript):
                astAssignee: ast.Name = stmt.targets[0]
                argData_dtypeName = hackSSOTdatatype(astAssignee.id)
                allImports.addImportFromStr(moduleConstructor, argData_dtypeName)
                astSubscript: ast.Subscript = stmt.value
                if isinstance(astSubscript.value, ast.Name) and astSubscript.value.id == astArg.arg and isinstance(astSubscript.slice, ast.Attribute):
                    indexAs_astAttribute: ast.Attribute = astSubscript.slice
                    indexAsStr = ast.unparse(indexAs_astAttribute)
                    argDataSlice = argData[eval(indexAsStr)]

                    onlyDataRLE = makeStrRLEcompacted(argDataSlice)
                    astStatement = cast(ast.Expr, ast.parse(onlyDataRLE).body[0])
                    dataAst = astStatement.value

                    arrayCall = ast.Call(
                        func=ast.Name(id=constructorName, ctx=ast.Load()) , args=[dataAst]
                        , keywords=[ast.keyword(arg='dtype', value=ast.Name(id=argData_dtypeName, ctx=ast.Load()) ) ] )

                    assignment = ast.Assign( targets=[astAssignee], value=arrayCall )
                    FunctionDefTarget.body.insert(0, assignment)
                    FunctionDefTarget.body.remove(stmt)

    FunctionDefTarget.args.args.remove(astArg)
    return FunctionDefTarget

def evaluate_argIn_body(FunctionDefTarget: ast.FunctionDef, astArg: ast.arg, argData: numpy.ndarray, Z0Z_listChaff: List[str]) -> ast.FunctionDef:
    moduleConstructor = datatypeModuleScalar
    for stmt in FunctionDefTarget.body.copy():
        if isinstance(stmt, ast.Assign):
            if isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Subscript):
                astAssignee: ast.Name = stmt.targets[0]
                argData_dtypeName = hackSSOTdatatype(astAssignee.id)
                allImports.addImportFromStr(moduleConstructor, argData_dtypeName)
                astSubscript: ast.Subscript = stmt.value
                if isinstance(astSubscript.value, ast.Name) and astSubscript.value.id == astArg.arg and isinstance(astSubscript.slice, ast.Attribute):
                    indexAs_astAttribute: ast.Attribute = astSubscript.slice
                    indexAsStr = ast.unparse(indexAs_astAttribute)
                    argDataSlice: int = argData[eval(indexAsStr)].item()
                    astCall = ast.Call(func=ast.Name(id=argData_dtypeName, ctx=ast.Load()) , args=[ast.Constant(value=argDataSlice)], keywords=[])
                    assignment = ast.Assign(targets=[astAssignee], value=astCall)
                    if astAssignee.id not in Z0Z_listChaff:
                        FunctionDefTarget.body.insert(0, assignment)
                    FunctionDefTarget.body.remove(stmt)
    FunctionDefTarget.args.args.remove(astArg)
    return FunctionDefTarget

def evaluateAnnAssignIn_body(FunctionDefTarget: ast.FunctionDef) -> ast.FunctionDef:
    moduleConstructor = datatypeModuleScalar
    for stmt in FunctionDefTarget.body.copy():
        if isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and isinstance(stmt.value, ast.Constant):
                astAssignee: ast.Name = stmt.target
                argData_dtypeName = hackSSOTdatatype(astAssignee.id)
                allImports.addImportFromStr(moduleConstructor, argData_dtypeName)
                astCall = ast.Call(func=ast.Name(id=argData_dtypeName, ctx=ast.Load()) , args=[stmt.value], keywords=[])
                assignment = ast.Assign(targets=[astAssignee], value=astCall)
                FunctionDefTarget.body.insert(0, assignment)
                FunctionDefTarget.body.remove(stmt)
    return FunctionDefTarget

def removeIdentifierFrom_body(FunctionDefTarget: ast.FunctionDef, astArg: ast.arg) -> ast.FunctionDef:
    for stmt in FunctionDefTarget.body.copy():
        if isinstance(stmt, ast.Assign):
            if isinstance(stmt.targets[0], ast.Subscript) and isinstance(stmt.targets[0].value, ast.Name):
                if stmt.targets[0].value.id == astArg.arg:
                    FunctionDefTarget.body.remove(stmt)
    FunctionDefTarget.args.args.remove(astArg)
    return FunctionDefTarget

def astObjectToAstConstant(FunctionDefTarget: ast.FunctionDef, object: str, value: int) -> ast.FunctionDef:
    """
    Replaces nodes in astFunction matching the AST of the string `object`
    with a constant node holding the provided value.
    """
    targetExpression = ast.parse(object, mode='eval').body
    targetDump = ast.dump(targetExpression, annotate_fields=False)

    def findNode(node: ast.AST) -> bool:
        return ast.dump(node, annotate_fields=False) == targetDump

    def replaceWithConstant(node: ast.AST) -> ast.AST:
        return ast.copy_location(ast.Constant(value=value), node)

    transformer = NodeReplacer(findNode, replaceWithConstant)
    newFunction = cast(ast.FunctionDef, transformer.visit(FunctionDefTarget))
    ast.fix_missing_locations(newFunction)
    return newFunction

def astNameToAstConstant(FunctionDefTarget: ast.FunctionDef, name: str, value: int) -> ast.FunctionDef:
    def findName(node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id == name

    def replaceWithConstant(node: ast.AST) -> ast.AST:
        return ast.copy_location(ast.Constant(value=value), node)

    return cast(ast.FunctionDef, NodeReplacer(findName, replaceWithConstant).visit(FunctionDefTarget))

def makeLauncherJobNumba(callableTarget: str, pathFilenameFoldsTotal: pathlib.Path) -> ast.Module:
    """Creates AST nodes for executing and recording results."""
    linesLaunch = f"""
if __name__ == '__main__':
    foldsTotal = {callableTarget}()
    print(foldsTotal)
    writeStream = open('{pathFilenameFoldsTotal.as_posix()}', 'w')
    writeStream.write(str(foldsTotal))
    writeStream.close()
"""
    return ast.parse(linesLaunch)

def addReturnJobNumba(FunctionDefTarget: ast.FunctionDef, stateJob: computationState) -> ast.FunctionDef:
    """Add multiplication and return statement to function, properly constructing AST nodes."""
    # Create AST for multiplication operation
    multiplicand = 'groupsOfFolds'
    datatype = hackSSOTdatatype(multiplicand)
    multiplyOperation = ast.BinOp(
        left=ast.Name(id=multiplicand, ctx=ast.Load()),
        op=ast.Mult(), right=ast.Constant(value=int(stateJob['foldGroups'][-1])))

    returnStatement = ast.Return(value=multiplyOperation)

    # This seems unnecessary; AI autocomplete: but it's a good habit to copy location info
    # Add source location info
    ast.copy_location(returnStatement, FunctionDefTarget.body[-1])

    # Set function return type annotation if not present
    allImports.addImportFromStr(datatypeModuleScalar, datatype)
    FunctionDefTarget.returns = ast.Name(id=datatype, ctx=ast.Load())

    # Add return statement to function body
    FunctionDefTarget.body.append(returnStatement)

    return FunctionDefTarget

def writeJobNumba(listDimensions: Sequence[int], callableTarget: str, algorithmSource: ModuleType, parametersNumba: Optional[ParametersNumba]=None, pathFilenameWriteJob: Optional[Union[str, os.PathLike[str]]] = None, **keywordArguments: Optional[Any]) -> pathlib.Path:
    """ Parameters: **keywordArguments: most especially for `computationDivisions` if you want to make a parallel job. Also `CPUlimit`. """
    """Notes about the existing logic:
    - the synthesized module must run well as a standalone interpreted Python script
    - `writeJobNumba` synthesizes a parameter-specific module by starting with code synthesized by `makeNumbaOptimizedFlow`, which improves the optimization
    - similarly, `writeJobNumba` should be a solid foundation for more optimizations, most especially compiling to a standalone executable, but the details of the next optimization step are unknown
    - the minimum runtime (on my computer) to compute a value unknown to mathematicians is 26 hours, therefore, we ant to ensure the value is seen by the user, but we must have ultra-light overhead.
    - perf_counter is for testing. When I run a real job, I delete those lines
    - avoid `with` statement
    """
    stateJob = makeStateJob(listDimensions, writeJob=False, **keywordArguments)
    pythonSource = inspect.getsource(algorithmSource)
    astModule = ast.parse(pythonSource)
    global allImports # TODO remove
    allImports = UniversalImportTracker()

    for statement in astModule.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            allImports.addAst(statement)

    FunctionDefTarget = next((node for node in astModule.body if isinstance(node, ast.FunctionDef) and node.name == callableTarget), None)

    if not FunctionDefTarget:
        raise ValueError(f"Could not find function {callableTarget} in source code.")

    # Define argument processing rules
    argumentRules = [
        (lambda arg: arg.arg in ['connectionGraph', 'gapsWhere'],
        lambda node, arg: moveArrayTo_body(node, arg, stateJob[arg.arg])),

        (lambda arg: arg.arg in ['track'],
        lambda node, arg: evaluateArrayIn_body(node, arg, stateJob[arg.arg])),

        (lambda arg: arg.arg in ['my'],
        lambda node, arg: evaluate_argIn_body(node, arg, stateJob[arg.arg], ['taskIndex', 'dimensionsTotal'])),

        (lambda arg: arg.arg in ['foldGroups'],
        lambda node, arg: removeIdentifierFrom_body(node, arg))
    ]

    # Process arguments using ArgumentProcessor
    argumentProcessor = ArgumentProcessor(argumentRules)
    FunctionDefTarget = argumentProcessor.process(FunctionDefTarget)

    FunctionDefTarget = evaluateAnnAssignIn_body(FunctionDefTarget)
    FunctionDefTarget = astNameToAstConstant(FunctionDefTarget, 'dimensionsTotal', int(stateJob['my'][indexMy.dimensionsTotal]))
    FunctionDefTarget = astObjectToAstConstant(FunctionDefTarget, 'foldGroups[-1]', int(stateJob['foldGroups'][-1]))

    FunctionDefTarget = addReturnJobNumba(FunctionDefTarget, stateJob)

    FunctionDefTarget, allImports = makeDecoratorJobNumba(FunctionDefTarget, allImports, parametersNumba)

    pathFilenameFoldsTotal = getPathFilenameFoldsTotal(stateJob['mapShape'])
    astLauncher = makeLauncherJobNumba(FunctionDefTarget.name, pathFilenameFoldsTotal)

    astImports = allImports.makeListAst()

    astModule = ast.Module(
        body=cast(List[ast.stmt]
                , astImports
                + [FunctionDefTarget]
                + [astLauncher])
        , type_ignores=[]
    )
    ast.fix_missing_locations(astModule)

    pythonSource = ast.unparse(astModule)
    pythonSource = autoflake.fix_code(pythonSource, ['mapFolding', 'numba', 'numpy'])

    if pathFilenameWriteJob is None:
        filename = getFilenameFoldsTotal(stateJob['mapShape'])
        pathRoot = getPathJobRootDEFAULT()
        pathFilenameWriteJob = pathlib.Path(pathRoot, pathlib.Path(filename).stem, pathlib.Path(filename).with_suffix('.py'))
    else:
        pathFilenameWriteJob = pathlib.Path(pathFilenameWriteJob)
    pathFilenameWriteJob.parent.mkdir(parents=True, exist_ok=True)

    pathFilenameWriteJob.write_text(pythonSource)
    return pathFilenameWriteJob

if __name__ == '__main__':
    setDatatypeModule('numpy', sourGrapes=True)
    setDatatypeFoldsTotal('int64', sourGrapes=True)
    setDatatypeElephino('uint8', sourGrapes=True)
    setDatatypeLeavesTotal('uint8', sourGrapes=True)
    listCallablesInline: List[str] = ['countInitialize', 'countParallel', 'countSequential']
    datatypeModuleScalar = 'numba'
    datatypeModuleDecorator = 'numba'
    decoratorModule = datatypeModuleDecorator
    decoratorCallable = 'jit'
    callableDispatcher = 'doTheNeedful'
    makeNumbaOptimizedFlow(listCallablesInline, callableDispatcher)

    listDimensions = [3,4]
    setDatatypeFoldsTotal('int64', sourGrapes=True)
    setDatatypeElephino('uint8', sourGrapes=True)
    setDatatypeLeavesTotal('uint8', sourGrapes=True)
    from mapFolding.syntheticModules import numba_countSequential
    algorithmSource: ModuleType = numba_countSequential
    datatypeModuleScalar = 'numba'
    datatypeModuleDecorator = 'numba'
    decoratorModule = datatypeModuleDecorator
    decoratorCallable = 'jit'
    pathFilenameModule = writeJobNumba(listDimensions, 'countSequential', algorithmSource, parametersNumbaDEFAULT)
