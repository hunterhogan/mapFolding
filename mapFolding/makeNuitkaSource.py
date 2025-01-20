from pathlib import Path
from pickle import loads
from typing import Final
import numpy

"""
Section: configure every time"""

# TODO configure this
mapShapeString = '[3x3x3]'
# NOTE ^^^^^^ pay attention

"""
Section: settings"""

pathJobs = Path('/apps/mapFolding/mapFolding/jobs')
pathThisJob = pathJobs / mapShapeString

pathFilenameAlgorithm = Path('/apps/mapFolding/mapFolding/countSequentialNoNumba.py')
pathFilenameData = pathThisJob / 'stateJob.pkl'
pathFilenameDestination = Path(f"/apps/mapFolding/nn/{mapShapeString}.py")

"""
Section: did you handle and include this stuff?"""

lineImportNumPy = "import numpy"
linePrintFoldsTotal = "print(foldsSubTotals.sum().item())"
linesAlgorithm = """"""
linesData = """"""
settingsNuitkaProject=f"""
# nuitka-project: --mode=onefile
# nuitka-project: --onefile-no-compression
# nuitka-project: --lto=yes
# nuitka-project: --clang
# nuitka-project: --output-dir={pathThisJob}
# nuitka-project: --output-filename={mapShapeString}.exe
"""
# nuitka-project:
"""
Section: do the work"""

WTFamIdoing = pathFilenameAlgorithm.read_text()
for lineSource in WTFamIdoing.splitlines():
    ImaIndent = '    '
    if lineSource.startswith(ImaIndent):
        lineSource = lineSource[len(ImaIndent):None]
    elif lineSource.startswith('#'):
        continue
    elif not lineSource:
        continue
    elif lineSource.startswith('def '):
        continue
    else:
        raise NotImplementedError("You didn't anticipate this.")
    linesAlgorithm = "\n".join([linesAlgorithm
                            , lineSource
                            ])

stateJob = loads(pathFilenameData.read_bytes())
connectionGraph: Final[numpy.ndarray] = stateJob['connectionGraph']
foldsSubTotals: numpy.ndarray = stateJob['foldsSubTotals']
gapsWhere: numpy.ndarray = stateJob['gapsWhere']
my: numpy.ndarray = stateJob['my']
the: numpy.ndarray = stateJob['the']
track: numpy.ndarray = stateJob['track']

pathFilenameFoldsTotal = stateJob['pathFilenameFoldsTotal']
lineDataPathFilenameFoldsTotal = "pathFilenameFoldsTotal = r'" + str(pathFilenameFoldsTotal) + "'\n"

def archivistFormatsArrayToCode(arrayTarget: numpy.ndarray, identifierName: str) -> str:
    """Format numpy array into a code string that recreates the array."""
    arrayAsTypeStr = numpy.array2string(
        arrayTarget,
        threshold=10000,
        max_line_width=100,
        separator=','
        )
    return f"{identifierName} = numpy.array({arrayAsTypeStr}, dtype=numpy.{arrayTarget.dtype})\n"

linesData = "\n".join([linesData
            , lineDataPathFilenameFoldsTotal
            , archivistFormatsArrayToCode(the, 'the')
            , archivistFormatsArrayToCode(my, 'my')
            , archivistFormatsArrayToCode(foldsSubTotals, 'foldsSubTotals')
            , archivistFormatsArrayToCode(gapsWhere, 'gapsWhere')
            , archivistFormatsArrayToCode(connectionGraph, 'connectionGraph')
            , archivistFormatsArrayToCode(track, 'track')
            ])

linesAll = "\n".join([settingsNuitkaProject
            , lineImportNumPy
            , linesData
            , linesAlgorithm
            , linePrintFoldsTotal
            ])

pathFilenameDestination.write_text(linesAll)
