"""Create a python module hardcoded to compute a map's foldsTotal.
- NumPy ndarray.
- Numba optimized.
- Absolutely no other imports.

Can create LLVM IR from the module: of unknown utility.
"""
# from mapFolding import dtypeDefault, dtypeSmall
from mapFolding import make_dtype, datatypeLarge, dtypeLarge
from mapFolding.someAssemblyRequired.inlineAfunction import Z0Z_inlineMapFolding
from mapFolding.someAssemblyRequired.jobsAndTasks import Z0Z_makeJob
import importlib
import llvmlite.binding
import numpy
import pathlib
import pickle
import python_minifier

listDimensions = [5,5]

# NOTE this overwrites files
Z0Z_inlineMapFolding()

identifierCallableLaunch = "goGoGadgetAbsurdity"

def convertNDArrayToStr(arrayTarget: numpy.ndarray, identifierName: str) -> str:
    arrayAsTypeStr = numpy.array2string(arrayTarget, threshold=100000, max_line_width=200, separator=',')
    stringMinimized = python_minifier.minify(arrayAsTypeStr)
    commaZeroMaximum = arrayTarget.shape[-1] - 1
    stringMinimized = stringMinimized.replace('[0' + ',0'*commaZeroMaximum + ']', '[0]*'+str(commaZeroMaximum+1))
    for countZeros in range(commaZeroMaximum, 2, -1):
        stringMinimized = stringMinimized.replace(',0'*countZeros + ']', ']+[0]*'+str(countZeros))
    return f"{identifierName} = numpy.array({stringMinimized}, dtype=numpy.{arrayTarget.dtype})"

def writeModuleWithNumba(listDimensions):
    numpy_dtypeLarge = dtypeLarge
    # numpy_dtypeDefault = dtypeDefault
    datatypeDefault = 'uint8'
    numpy_dtypeDefault = make_dtype(datatypeDefault)
    numpy_dtypeSmall = numpy_dtypeDefault
    # forceinline=True might actually be useful
    parametersNumba = f"numba.types.{datatypeLarge}(), \
cache=True, \
nopython=True, \
fastmath=True, \
forceinline=True, \
inline='always', \
looplift=False, \
_nrt=True, \
error_model='numpy', \
parallel=False, \
boundscheck=False, \
no_cfunc_wrapper=False, \
no_cpython_wrapper=False, \
"
# no_cfunc_wrapper=True, \
# no_cpython_wrapper=True, \

    pathFilenameData = Z0Z_makeJob(listDimensions, datatypeDefault=numpy_dtypeDefault, datatypeLarge=numpy_dtypeLarge, datatypeSmall=numpy_dtypeSmall)

    pathFilenameAlgorithm = pathlib.Path('/apps/mapFolding/mapFolding/someAssemblyRequired/countSequentialNoNumba.py')
    pathFilenameDestination = pathFilenameData.with_stem(pathFilenameData.parent.name).with_suffix(".py")

    lineNumba = f"@numba.jit({parametersNumba})"

    linesImport = "\n".join([
                        "import numpy"
                        , "import numba"
                        ])

    stateJob = pickle.loads(pathFilenameData.read_bytes())

    ImaIndent = '    '
    linesDataDynamic = """"""
    linesDataDynamic = "\n".join([linesDataDynamic
            , ImaIndent + f"foldsTotal = numba.types.{datatypeLarge}(0)"
            , ImaIndent + convertNDArrayToStr(stateJob['my'], 'my')
            , ImaIndent + convertNDArrayToStr(stateJob['foldGroups'], 'foldGroups')
            , ImaIndent + convertNDArrayToStr(stateJob['gapsWhere'], 'gapsWhere')
            , ImaIndent + convertNDArrayToStr(stateJob['track'], 'track')
            ])

    linesDataStatic = """"""
    linesDataStatic = "\n".join([linesDataStatic
            , ImaIndent + convertNDArrayToStr(stateJob['connectionGraph'], 'connectionGraph')
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

    linesLaunch = """"""
    linesLaunch = linesLaunch + f"""
if __name__ == '__main__':
    import time
    timeStart = time.perf_counter()
    {identifierCallableLaunch}()
    print(time.perf_counter() - timeStart)"""

    linesWriteFoldsTotal = """"""
    linesWriteFoldsTotal = "\n".join([linesWriteFoldsTotal
                                    , "    foldsTotal = foldGroups[0:-1].sum() * foldGroups[-1]"
                                    , "    print(foldsTotal)"
                                    , "    with numba.objmode():"
                                    , f"        open('{pathFilenameFoldsTotal.as_posix()}', 'w').write(str(foldsTotal))"
                                    , "    return foldsTotal"
                                    ])

    linesAll = "\n".join([
                        linesImport
                        , linesAlgorithm
                        , linesWriteFoldsTotal
                        , linesLaunch
                        ])

    pathFilenameDestination.write_text(linesAll)

    return pathFilenameDestination

def writeModuleLLVM(pathFilenamePythonFile: pathlib.Path) -> pathlib.Path:
    pathRootPackage = pathlib.Path('c:/apps/mapFolding')
    relativePathModule = pathFilenamePythonFile.relative_to(pathRootPackage)
    moduleTarget = '.'.join(relativePathModule.parts)[0:-len(relativePathModule.suffix)]
    moduleTargetImported = importlib.import_module(moduleTarget)
    linesLLVM = moduleTargetImported.__dict__[identifierCallableLaunch].inspect_llvm()[()]
    moduleLLVM = llvmlite.binding.module.parse_assembly(linesLLVM)
    pathFilenameLLVM = pathFilenamePythonFile.with_suffix(".ll")
    pathFilenameLLVM.write_text(str(moduleLLVM))
    return pathFilenameLLVM

if __name__ == '__main__':
    pathFilenamePythonFile = writeModuleWithNumba(listDimensions)
    pathFilenameLLVM = writeModuleLLVM(pathFilenamePythonFile)
