from numpy import arange as A
from numpy import array as a
from numpy import ravel as R
from numpy import concatenate as C
import itertools
import more_itertools
import numpy
import numpy as np
import pathlib
import pickle
import python_minifier
pathFilenameData = pathlib.Path("/apps/mapFolding/mapFolding/jobs/p2x2x2x2x2x2/stateJob.pkl")
stateJob = pickle.loads(pathFilenameData.read_bytes())
connectionGraph: numpy.ndarray = stateJob['connectionGraph']
dictionaryLengths = {}
# https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi

# The baseline function
def Z00convertNDArrayToStr(arrayTarget: numpy.ndarray, identifierName: str) -> str:
    arrayAsTypeStr = numpy.array2string(arrayTarget, threshold=1000000, max_line_width=200, separator=',')
    return f"{identifierName} = numpy.array({arrayAsTypeStr}, dtype=numpy.{arrayTarget.dtype})"

def Z01convertNDArrayToStr(arrayTarget: numpy.ndarray, identifierName: str) -> str:
    arrayAsTypeStr = numpy.array2string(arrayTarget, threshold=1000000, max_line_width=200, separator=',')
    stringMinimized = python_minifier.minify(arrayAsTypeStr)
    commaZeroMaximum = arrayTarget.shape[-1] - 1
    stringMinimized = stringMinimized.replace('[0' + ',0'*commaZeroMaximum + ']', '[0]*'+str(commaZeroMaximum+1))
    for countZeros in range(commaZeroMaximum, 2, -1):
        stringMinimized = stringMinimized.replace(',0'*countZeros + ']', ']+[0]*'+str(countZeros))
    return f"{identifierName} = numpy.array({stringMinimized}, dtype=numpy.{arrayTarget.dtype})"

def Z02convertNDArrayToStr(arrayTarget: numpy.ndarray, identifierName: str):
    shapeArray = arrayTarget.shape
    arrayAsNestedLists = arrayTarget[:,:].copy().tolist()
    for axis0, axis1 in itertools.product(range(shapeArray[0]), range(shapeArray[1])):
        ImaList = arrayTarget[axis0, axis1].copy().tolist()
        listWithRanges = []
        for group in more_itertools.consecutive_groups(ImaList):
            ImaSerious = list(group)
            if len(ImaSerious) <= 4:
                listWithRanges += ImaSerious
            else:
                ImaRange = [range(ImaSerious[0],ImaSerious[-1]+1)]
                listWithRanges += ImaRange
        arrayAsNestedLists[axis0][axis1] = listWithRanges

    arrayAsTypeStr = str(arrayAsNestedLists)
    stringMinimized = python_minifier.minify(arrayAsTypeStr)
    commaZeroMaximum = arrayTarget.shape[-1] - 1
    stringMinimized = stringMinimized.replace('[0' + ',0'*commaZeroMaximum + ']', '[0]*'+str(commaZeroMaximum+1))
    for countZeros in range(commaZeroMaximum, 2, -1):
        stringMinimized = stringMinimized.replace(',0'*countZeros + ']', ']+[0]*'+str(countZeros))

    stringMinimized = stringMinimized.replace('range', '*range')

    return f"{identifierName} = numpy.array({stringMinimized}, dtype=numpy.{arrayTarget.dtype})"

# def Z03convertNDArrayToStr(arrayTarget: numpy.ndarray, identifierName: str) -> str:
#     stringMinimized = python_minifier.minify(str(list(more_itertools.run_length.encode(arrayTarget.tolist()))))
#     return f"{identifierName} = numpy.array({stringMinimized}, dtype=numpy.{arrayTarget.dtype})"

COUNTfunctions = 4

for id in range(COUNTfunctions):
    idAsStr = f'Z{str(id).zfill(2)}'
    callableComparand = globals()[f'{idAsStr}convertNDArrayToStr']

    # stringReturned = python_minifier.minify(callableComparand(connectionGraph, idAsStr))
    stringReturned = callableComparand(connectionGraph, idAsStr)
    if id == 3:
        print(stringReturned)
    exec(stringReturned)
    assert numpy.array_equal(connectionGraph, globals()[idAsStr])
    dictionaryLengths[idAsStr] = len(stringReturned)

for idAsStr, lengthString in sorted(dictionaryLengths.items(), key=lambda item: item[1]):
    print(f"{idAsStr}: {lengthString}")
