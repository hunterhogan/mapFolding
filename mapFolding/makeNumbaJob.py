"""Create a python module hardcoded to compute a map's foldsTotal.
NumPy ndarray.
Numba optimized.
Absolutely no other imports.
"""
from mapFolding import datatypeLarge, dtypeLarge, dtypeDefault
from mapFolding.inlineAfunction import Z0Z_inlineMapFolding
from mapFolding.startHere import Z0Z_makeJob
import ast
import llvmlite.binding as llvm
import numba
import numba.core.compiler
import numba.core.registry
import numpy
import pathlib
import pickle

listDimensions = [6,6]

# NOTE this overwrites files
Z0Z_inlineMapFolding()

identifierCallableLaunch = "goGoGadgetAbsurdity"

def archivistFormatsArrayToCode(arrayTarget: numpy.ndarray, identifierName: str) -> str:
    """Format numpy array into a code string that recreates the array."""
    arrayAsTypeStr = numpy.array2string(arrayTarget, threshold=10000, max_line_width=200, separator=',')
    return f"{identifierName} = numpy.array({arrayAsTypeStr}, dtype=numpy.{arrayTarget.dtype})"

def makePythonCode(listDimensions):
    numpy_dtypeLarge = dtypeLarge
    numpy_dtypeDefault = dtypeDefault

    parametersNumba = f"numba.types.{datatypeLarge}(), parallel=False, boundscheck=False, error_model='numpy', fastmath=True, nogil=True, nopython=True"

    pathFilenameData = Z0Z_makeJob(listDimensions, datatypeDefault=numpy_dtypeDefault, datatypeLarge=numpy_dtypeLarge)

    pathFilenameAlgorithm = pathlib.Path('/apps/mapFolding/mapFolding/countSequentialNoNumba.py')
    pathFilenameDestination = pathFilenameData.with_stem(pathFilenameData.parent.name).with_suffix(".py")

    lineNumba = f"@numba.cfunc({parametersNumba})"

    linesImport = "\n".join([
                        "import numpy"
                        , "import numba"
                        ])

    stateJob = pickle.loads(pathFilenameData.read_bytes())

    ImaIndent = '    '
    linesDataDynamic = """"""
    linesDataDynamic = "\n".join([linesDataDynamic
            , ImaIndent + archivistFormatsArrayToCode(stateJob['my'], 'my')
            , ImaIndent + archivistFormatsArrayToCode(stateJob['foldsSubTotals'], 'foldsSubTotals')
            , ImaIndent + archivistFormatsArrayToCode(stateJob['gapsWhere'], 'gapsWhere')
            , ImaIndent + archivistFormatsArrayToCode(stateJob['track'], 'track')
            ])

    linesDataStatic = """"""
    linesDataStatic = "\n".join([linesDataStatic
            , ImaIndent + archivistFormatsArrayToCode(stateJob['the'], 'the')
            , ImaIndent + archivistFormatsArrayToCode(stateJob['connectionGraph'], 'connectionGraph')
            ])

    pathFilenameFoldsTotal: pathlib.Path = stateJob['pathFilenameFoldsTotal']

    linesAlgorithm = """"""
    for lineSource in pathFilenameAlgorithm.read_text().splitlines():
        if lineSource.startswith('#'):
            continue
        elif not lineSource:
            continue
        elif lineSource.startswith('def '):
            lineSource = "\n".join([lineNumba
                                , f"def {identifierCallableLaunch}():"
                                , linesDataDynamic
                                , linesDataStatic
                                ])
        linesAlgorithm = "\n".join([linesAlgorithm
                            , lineSource
                            ])

    lineReturn = f"{ImaIndent}return foldsSubTotals.sum().item()"

    linesPythonCode = "\n".join([
            linesImport
            , linesAlgorithm
            , lineReturn
            ])

    return pathFilenameDestination,pathFilenameFoldsTotal,linesPythonCode

def numbaTransformation(pythonCode: str) -> str:
    """Transform Numba-optimized Python code into LLVM IR code"""
    # Initialize LLVM
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    # Parse the Python code to get function object without executing
    tree = ast.parse(pythonCode)
    compiled = compile(tree, '<string>', 'exec')
    namespace = {}
    eval(compiled, namespace)
    compiled_function = namespace[identifierCallableLaunch]

    # Get the compilation result
    contextTyping = numba.core.registry.cpu_target.typing_context
    contextTarget = numba.core.registry.cpu_target.target_context

    compilationResult = numba.core.compiler.compile_extra(
        func=compiled_function._pyfunc,
        args=((),),
        return_type=eval(f"numba.types.{datatypeLarge}"),
        flags={},
        locals={},
        typingctx=contextTyping,
        targetctx=contextTarget)

    # Get the LLVM IR
    llvmModule = str(compilationResult.library.get_llvm_str())

    # Create new code that uses LLVM directly
    llvmCode = f"""from llvmlite import binding as llvm

# Initialize LLVM
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

# Create module from IR
module = llvm.parse_assembly('''
{llvmModule}
''')

# Create execution engine
target = llvm.Target.from_default_triple()
target_machine = target.create_target_machine()
execution_engine = llvm.create_execution_engine(target_machine)

# Get function
function_address = execution_engine.get_function_address("{identifierCallableLaunch}")

# Create callable
from ctypes import CFUNCTYPE, c_{datatypeLarge}
compiled_function = CFUNCTYPE(c_{datatypeLarge})(function_address)

if __name__ == '__main__':
    foldsTotal = compiled_function()
"""
    return llvmCode

pathFilenameWritePythonFile, pathFilenameFoldsTotal, linesPythonCode = makePythonCode(listDimensions)
linesLLVMCode = numbaTransformation(linesPythonCode)

linesLaunch = """"""
linesLaunch = linesLaunch + f"""
if __name__ == '__main__':
    foldsTotal = {identifierCallableLaunch}()"""

linesWriteFoldsTotal = """"""
linesWriteFoldsTotal = "\n".join([linesWriteFoldsTotal
                                , "    print(foldsTotal)"
                                , f"    open('{pathFilenameFoldsTotal.as_posix()}', 'w').write(str(foldsTotal))"
                                ])

linesAll = "\n".join([
                    linesLLVMCode
                    , linesWriteFoldsTotal
                    ])

pathFilenameWritePythonFile.write_text(linesAll)
