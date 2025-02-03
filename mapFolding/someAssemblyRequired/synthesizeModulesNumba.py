from mapFolding import EnumIndices, relativePathSyntheticModules, setDatatypeElephino, setDatatypeFoldsTotal, setDatatypeLeavesTotal, setDatatypeModule
from mapFolding import indexMy, indexTrack, getAlgorithmSource, ParametersNumba, parametersNumbaDEFAULT, hackSSOTdatatype, hackSSOTdtype
from typing import cast, Dict, List, Optional, Sequence, Set, Type, Union
from types import ModuleType
import ast
import inspect
import numba
import numpy
import pathlib

"""TODO
Convert types
    e.g. `groupsOfFolds: int = 0` to `groupsOfFolds = numba.types.{datatypeLarge}(0)`
    This isn't necessary for Numba, but I may the infrastructure for other compilers or paradigms."""

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

    def visit_Call(self, callNode: ast.Call) -> ast.AST:
        callNodeVisited = self.generic_visit(callNode)
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

def decorateCallableWithNumba(astCallable: ast.FunctionDef, parallel: bool=False) -> ast.FunctionDef:
    """
    Decorates an AST function definition with Numba JIT compilation parameters.

    This function processes an AST FunctionDef node and adds Numba-specific decorators
    for JIT compilation. It handles array parameter typing and compilation options.

    Parameters
    ----------
    astCallable : ast.FunctionDef
        The AST node representing the function to be decorated with Numba JIT.
    parallel : bool, optional
        Whether to enable parallel execution in Numba compilation.
        Default is False.

    Returns
    -------
    ast.FunctionDef
        The modified AST function definition node with added Numba decorators.

    Notes
    -----
    The function performs the following main tasks:
    1. Processes function parameters to create Numba-compatible type signatures
    2. Constructs appropriate Numba compilation parameters
    3. Creates and attaches a @numba.jit decorator to the function
    Special handling is included for the 'countInitialize' function, which receives
    empty compilation parameters.
    The function relies on external parameters:
    - parametersNumbaDEFAULT: Default Numba compilation parameters
    - ParametersNumba: Class/type for handling Numba parameters
    - hackSSOTdatatype: Function for determining default datatypes
    """
    def makeNumbaParameterSignatureElement(signatureElement: ast.arg):
        """
        Converts an AST function parameter signature element into a Numba-compatible type annotation.

        This function processes parameter annotations for array types, handling both shape and datatype
        specifications. It supports multi-dimensional arrays through tuple-based shape definitions and
        various numeric datatypes.

        Parameters
        ----------
        signatureElement : ast.arg
            The AST argument node containing the parameter's name and type annotation.
            Expected annotation format: Type[shape_tuple, dtype]
            where shape_tuple can be either a single dimension or a tuple of dimensions,
            and dtype specifies the data type.

        Returns
        -------
        ast.Subscript
            A Numba-compatible type annotation as an AST node, representing an array type
            with the specified shape and datatype.

        Notes
        -----
        The function handles two main cases for shape specifications:
        1. Multi-dimensional arrays with tuple-based shapes
        2. Single-dimension arrays with simple slice notation
        The datatype can be either explicitly specified in the annotation or determined
        through a fallback mechanism using hackSSOTdatatype().
        """
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

            datatypeNumba = ast.Attribute(value=ast.Name(id='numba', ctx=ast.Load()), attr=datatype_attr, ctx=ast.Load())

            return ast.Subscript(value=datatypeNumba, slice=shapeAST, ctx=ast.Load())

    # TODO: more explicit handling of decorators. I'm able to ignore this because I know `algorithmSource` doesn't have any decorators.
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

def inlineOneCallable(codeSource, callableTarget):
    """
    Inlines a target callable function and its dependencies within the provided code source.

    This function performs function inlining, optionally adds Numba decorators, and handles array access unpacking
    for specific callable targets. It processes the source code through AST manipulation and returns the modified source.

    Parameters:
        codeSource (str): The source code containing the callable to be inlined.
        callableTarget (str): The name of the callable function to be inlined. Special handling is provided for
            'countParallel', 'countInitialize', and 'countSequential'.

    Returns:
        str: The modified source code with the inlined callable and necessary imports.

    The function performs the following operations:
    1. Parses the source code into an AST
    2. Extracts import statements and function definitions
    3. Inlines the target function using RecursiveInliner
    4. Applies Numba decoration if needed
    5. Handles special array access unpacking for 'countSequential'
    6. Reconstructs and returns the modified source code

    Note:
        - Special handling is provided for 'countParallel', 'countInitialize', and 'countSequential' targets
        - For 'countSequential', additional array access unpacking is performed for 'my' and 'track' indices
        - `UnpackArrayAccesses` would need modification to handle 'countParallel'
    """

    codeParsed: ast.Module = ast.parse(codeSource, type_comments=True)
    codeSourceImportStatements = {statement for statement in codeParsed.body if isinstance(statement, (ast.Import, ast.ImportFrom))}
    dictionaryFunctions = {statement.name: statement for statement in codeParsed.body if isinstance(statement, ast.FunctionDef)}
    callableInlinerWorkhorse = RecursiveInliner(dictionaryFunctions)
    callableInlined = callableInlinerWorkhorse.inlineFunctionBody(callableTarget)

    if callableInlined:
        ast.fix_missing_locations(callableInlined)
        parallel = callableTarget == 'countParallel'
        callableDecorated = decorateCallableWithNumba(callableInlined, parallel)

        if callableTarget == 'countSequential':
            unpackerMy = UnpackArrayAccesses(indexMy, 'my')
            callableDecorated = cast(ast.FunctionDef, unpackerMy.visit(callableDecorated))
            ast.fix_missing_locations(callableDecorated)

            unpackerTrack = UnpackArrayAccesses(indexTrack, 'track')
            callableDecorated = cast(ast.FunctionDef, unpackerTrack.visit(callableDecorated))
            ast.fix_missing_locations(callableDecorated)

        moduleAST = ast.Module(body=cast(List[ast.stmt], list(codeSourceImportStatements) + [callableDecorated]), type_ignores=[])
        ast.fix_missing_locations(moduleAST)
        moduleSource = ast.unparse(moduleAST)
        return moduleSource

class AppendDunderInit(ast.NodeTransformer):
    """AST transformer that validates and appends imports to __init__.py files."""

    def __init__(self, listPathFilenamesDestination: list[tuple[pathlib.Path, str]]):
        self.listPathFilenamesDestination = listPathFilenamesDestination
        self.listTuplesDunderInit = []

    def process_init_files(self) -> list[tuple[pathlib.Path, str]]:
        for pathFilename, callableTarget in self.listPathFilenamesDestination:
            pathDunderInit = pathFilename.parent / "__init__.py"

            # Create empty init if doesn't exist
            if not pathDunderInit.exists():
                pathDunderInit.write_text("")

            # Parse existing init file
            try:
                treeInit = ast.parse(pathDunderInit.read_text())
            except SyntaxError:
                treeInit = ast.Module(body=[], type_ignores=[])

            # Compute the lowercase module target
            moduleTarget = "." + pathFilename.stem
            moduleTargetLower = moduleTarget.lower()

            # Track existing imports as (normalizedModule, name)
            setImportsExisting = set()
            for node in treeInit.body:
                if isinstance(node, ast.ImportFrom) and node.module:
                    # Compare on a lowercase basis
                    if node.module.lower() == moduleTargetLower:
                        for alias in node.names:
                            setImportsExisting.add((moduleTargetLower, alias.name))

            # Only append if this exact import doesn't exist
            if (moduleTargetLower, callableTarget) not in setImportsExisting:
                newImport = ast.ImportFrom(
                    module=moduleTarget,
                    names=[ast.alias(name=callableTarget, asname=None)],
                    level=0
                )
                treeInit.body.append(newImport)
                ast.fix_missing_locations(treeInit)
                pathDunderInit.write_text(ast.unparse(treeInit))

            self.listTuplesDunderInit.append((pathDunderInit, callableTarget))

        return self.listTuplesDunderInit

def inlineMapFoldingNumba(listCallablesAsStr: List[str], algorithmSource: Optional[ModuleType] = None):
    """Synthesizes numba-optimized versions of map folding functions.
    This function creates specialized versions of map folding functions by inlining
    target callables and generating optimized modules. It handles the code generation
    and file writing process.

    Parameters:
        listCallablesAsStr (List[str]): List of callable names to be processed as strings.
        algorithmSource (Optional[ModuleType], optional): Source module containing the algorithms.
            If None, will be obtained via getAlgorithmSource(). Defaults to None.

    Returns:
        List[Tuple[pathlib.Path, str]]: List of tuples containing:
            - Generated file paths
            - Associated callable names

    Raises:
        Exception: If inline operation fails during code generation.

    Note:
        - Generated files are placed in a synthetic modules subdirectory
        - Modifies __init__.py files to expose generated modules
        - Current implementation contains hardcoded paths that should be abstracted
    """
    if not algorithmSource:
        algorithmSource = getAlgorithmSource()

    listPathFilenamesDestination: list[tuple[pathlib.Path, str]] = []

    # TODO abstract this process
    # especially remove the hardcoded paths and filenames

    for callableTarget in listCallablesAsStr:
        codeSource = inspect.getsource(algorithmSource)
        moduleSource = inlineOneCallable(codeSource, callableTarget)
        if not moduleSource:
            raise Exception("Pylance, OMG! The sky is falling!")
        pathFilenameAlgorithm = pathlib.Path(inspect.getfile(algorithmSource))
        pathFilenameDestination = pathFilenameAlgorithm.parent / relativePathSyntheticModules / pathFilenameAlgorithm.with_stem("numba"+callableTarget[5:None]).name
        pathFilenameDestination.write_text(moduleSource)
        listPathFilenamesDestination.append((pathFilenameDestination, callableTarget))

    # This almost works: it duplicates existing imports, though
    listTuplesDunderInit = AppendDunderInit(listPathFilenamesDestination).process_init_files()

if __name__ == '__main__':
    listCallablesAsStr: List[str] = ['countInitialize', 'countParallel', 'countSequential']
    setDatatypeModule('numpy', sourGrapes=True)
    setDatatypeFoldsTotal('int64', sourGrapes=True)
    setDatatypeElephino('uint8', sourGrapes=True)
    setDatatypeLeavesTotal('uint8', sourGrapes=True)
    inlineMapFoldingNumba(listCallablesAsStr)
