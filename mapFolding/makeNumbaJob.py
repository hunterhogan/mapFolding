from mapFolding.startHere import Z0Z_makeJob
from mapFolding.inlineAfunction import Z0Z_inlineMapFolding
import numpy
import pathlib
import pickle

# TODO write foldsTotal to a file

"""
Section: configure every time"""

# NOTE this overwrites files
Z0Z_inlineMapFolding()

# TODO configure this
mapShape = [2,2]
# NOTE ^^^^^^ pay attention

"""
Section: settings"""

parametersNumba = "cache=True, parallel=False, boundscheck=False, error_model='numpy', fastmath=True, nogil=True, nopython=True"

pathFilenameData = Z0Z_makeJob(mapShape)
pathJob = pathFilenameData.parent
pathFilenameAlgorithm = pathlib.Path('/apps/mapFolding/mapFolding/countSequentialNoNumba.py')
pathFilenameDestination = pathFilenameData.with_stem(pathJob.name).with_suffix(".py")

identifierCallableLaunch = "goGoGadgetAbsurdity"

ImaIndent = '    '

"""
Section: did you handle and include this stuff?"""

lineNumba = f"@numba.jit({parametersNumba})"
linesWriteFoldsTotal = """"""
linesAlgorithm = """"""
linesDataDynamic = """"""
linesDataStatic = """"""
linesLaunch = """"""

"""
Section: do the work"""

linesImport = "\n".join([
                        "import numpy"
                        , "import numba"
                        ])

stateJob = pickle.loads(pathFilenameData.read_bytes())
connectionGraph: numpy.ndarray = stateJob['connectionGraph']
foldsSubTotals: numpy.ndarray = stateJob['foldsSubTotals']
gapsWhere: numpy.ndarray = stateJob['gapsWhere']
my: numpy.ndarray = stateJob['my']
the: numpy.ndarray = stateJob['the']
track: numpy.ndarray = stateJob['track']

pathFilenameFoldsTotal = stateJob['pathFilenameFoldsTotal']
lineDataPathFilenameFoldsTotal = "pathFilenameFoldsTotal = r'" + str(pathFilenameFoldsTotal) + "'\n"

def archivistFormatsArrayToCode(arrayTarget: numpy.ndarray, identifierName: str) -> str:
    """Format numpy array into a code string that recreates the array."""
    arrayAsTypeStr = numpy.array2string( arrayTarget, threshold=10000, max_line_width=200, separator=',' )
    return f"{identifierName} = numpy.array({arrayAsTypeStr}, dtype=numpy.{arrayTarget.dtype})\n"

linesDataDynamic = "\n".join([linesDataDynamic
            , ImaIndent + archivistFormatsArrayToCode(my, 'my')
            , ImaIndent + archivistFormatsArrayToCode(foldsSubTotals, 'foldsSubTotals')
            , ImaIndent + archivistFormatsArrayToCode(gapsWhere, 'gapsWhere')
            , ImaIndent + archivistFormatsArrayToCode(track, 'track')
            ])

linesDataStatic = "\n".join([linesDataStatic
            # , lineDataPathFilenameFoldsTotal
            , archivistFormatsArrayToCode(the, 'the')
            , archivistFormatsArrayToCode(connectionGraph, 'connectionGraph')
            ])

linesWriteFoldsTotal = "\n".join([linesWriteFoldsTotal
                                , f"{ImaIndent}print(foldsSubTotals.sum().item())"
                                ])

WTFamIdoing = pathFilenameAlgorithm.read_text()
for lineSource in WTFamIdoing.splitlines():
    if lineSource.startswith('#'):
        continue
    elif not lineSource:
        continue
    elif lineSource.startswith('def '):
        lineSource = lineNumba + "\n"
        lineSource = lineSource + f"def {identifierCallableLaunch}():\n"
        lineSource = lineSource + linesDataDynamic
    linesAlgorithm = "\n".join([linesAlgorithm
                            , lineSource
                            ])

linesLaunch = linesLaunch + f"""
if __name__ == '__main__':
    {identifierCallableLaunch}()

"""
linesAll = "\n".join([
            linesImport
            , linesDataStatic
            , linesAlgorithm
            , linesWriteFoldsTotal
            , linesLaunch
            ])

# from python_minifier import minify
# linesAll = minify(linesAll, hoist_literals=False, rename_globals=True)

pathFilenameDestination.write_text(linesAll)
