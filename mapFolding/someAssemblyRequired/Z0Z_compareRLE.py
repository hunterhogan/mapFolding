import itertools
import numpy
from numpy import repeat as R
import numpy as np
import pathlib
import pickle
import python_minifier
from numpy import arange as A

pathFilenameData = pathlib.Path("/apps/mapFolding/mapFolding/jobs/p2x2x2x2x2x2x2/stateJob.pkl")
stateJob = pickle.loads(pathFilenameData.read_bytes())
connectionGraph: numpy.ndarray = stateJob['connectionGraph']
dictionaryLengths = {}
# https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi

# The baseline function
def Z00convertNDArrayToStr(arrayTarget: numpy.ndarray, identifierName: str) -> str:
    arrayAsTypeStr = numpy.array2string(arrayTarget, threshold=1000000, max_line_width=200, separator=',')
    return f"{identifierName} = numpy.array({arrayAsTypeStr}, dtype=numpy.{arrayTarget.dtype})"

def Z01convertNDArrayToStr(arrayTarget: numpy.ndarray, identifierName: str) -> str:
    def rle(arrayInput):
        arrayLength = len(arrayInput)
        if arrayLength == 0:
            return (None, None, None)
        arrayDiff = arrayInput[1:] != arrayInput[:-1]
        arrayIndices = numpy.append(numpy.where(arrayDiff), arrayLength - 1)
        arrayRunLengths = numpy.diff(numpy.append(-1, arrayIndices))
        arrayPositions = numpy.cumsum(numpy.append(0, arrayRunLengths))[:-1]
        arrayValues = arrayInput[arrayIndices]
        return (arrayRunLengths, arrayPositions, arrayValues)
    arrayFlattened = arrayTarget.ravel()
    arrayRunLengths, arrayPositions, arrayValues = rle(arrayFlattened)
    arrayAsTypeStr = f"R({arrayValues.tolist()},{arrayRunLengths.tolist()}).reshape{arrayTarget.shape}" # type: ignore

    return f"{identifierName} = numpy.array({arrayAsTypeStr},dtype=numpy.{arrayTarget.dtype})"

def Z02convertNDArrayToStr(arrayTarget: numpy.ndarray, identifierName: str) -> str:
    """
    Run-length encoding for N-dimensional arrays
    """
    def rleAnyShape(arrayInput):
        shapeOriginal = arrayInput.shape
        arrayFlattened = arrayInput.ravel()
        arrayLength = len(arrayFlattened)

        if arrayLength == 0:
            return numpy.array([]), numpy.array([]), shapeOriginal

        arrayDiff = numpy.concatenate(([True], arrayFlattened[1:] != arrayFlattened[:-1], [True]))
        arrayIndices = numpy.nonzero(arrayDiff)[0]
        arrayRunLengths = numpy.diff(arrayIndices)
        arrayValues = arrayFlattened[arrayIndices[:-1]]

        return arrayRunLengths, arrayValues, shapeOriginal

    arrayRunLengths, arrayValues, shapeOriginal = rleAnyShape(arrayTarget)

    if len(arrayRunLengths) == 0:
        arrayAsTypeStr = "numpy.array([])"
    else:
        # Use nested compression if beneficial
        if len(arrayRunLengths) > len(arrayTarget) // 4:
            # If RLE isn't providing good compression, use direct representation
            arrayAsTypeStr = f"numpy.array({arrayTarget.tolist()})"
        else:
            arrayAsTypeStr = f"R({arrayValues.tolist()},{arrayRunLengths.tolist()}).reshape{shapeOriginal}"

    return f"{identifierName} = numpy.array({arrayAsTypeStr},dtype=numpy.{arrayTarget.dtype})"

def Z03convertNDArrayToStr(arrayTarget: numpy.ndarray, identifierName: str) -> str:
    arrayFlattened = arrayTarget.ravel()
    arrayValues = []
    arrayRunLengths = []
    for valueKey, groupValues in itertools.groupby(arrayFlattened):
        arrayValues.append(valueKey)
        arrayRunLengths.append(len(list(groupValues)))
    expressionDecoded = f"numpy.repeat({arrayValues},{arrayRunLengths}).reshape{arrayTarget.shape}"
    return f"{identifierName} = numpy.array({expressionDecoded},dtype=numpy.{arrayTarget.dtype})"

def Z07convertNDArrayToStr(arrayTarget: numpy.ndarray, identifierName: str) -> str:
    def runsOfOnesList(bits):
        return [sum(groupValues) for bitKey, groupValues in itertools.groupby(bits) if bitKey]
    arrayFlattened = arrayTarget.ravel()
    # Run-length logic for all values, not just ones:
    arrayValues = []
    arrayRunLengths = []
    for valueKey, groupValues in itertools.groupby(arrayFlattened):
        arrayValues.append(valueKey)
        arrayRunLengths.append(len(list(groupValues)))
    expressionDecoded = f"numpy.repeat({arrayValues},{arrayRunLengths}).reshape{arrayTarget.shape}"
    return f"{identifierName} = numpy.array({expressionDecoded},dtype=numpy.{arrayTarget.dtype})"
def Z04convertNDArrayToStr(arrayTarget: numpy.ndarray, identifierName: str) -> str:
    def runsOfOnesArray(bits):
        arrayBounded = numpy.hstack(([0], bits, [0]))
        arrayDiff = numpy.diff(arrayBounded)
        arrayStarts, = numpy.where(arrayDiff > 0)
        arrayEnds, = numpy.where(arrayDiff < 0)
        return arrayEnds - arrayStarts
    arrayFlattened = arrayTarget.ravel()
    # Same groupby approach to handle all values:
    arrayValues = []
    arrayRunLengths = []
    for valueKey, groupValues in itertools.groupby(arrayFlattened):
        arrayValues.append(valueKey)
        arrayRunLengths.append(len(list(groupValues)))
    expressionDecoded = f"numpy.repeat({arrayValues},{arrayRunLengths}).reshape{arrayTarget.shape}"
    return f"{identifierName} = numpy.array({expressionDecoded},dtype=numpy.{arrayTarget.dtype})"

def Z05convertNDArrayToStr(arrayTarget: numpy.ndarray, identifierName: str) -> str:
    # Merge chunks into one list before creating the array
    arrayFlattened = arrayTarget.ravel()
    arrayDtypeName = arrayTarget.dtype.name
    chunkSize = 50
    listStrings = []
    indexStart = 0
    while indexStart < len(arrayFlattened):
        lastIndex = indexStart + chunkSize
        subList = arrayFlattened[indexStart:lastIndex].tolist()
        listStrings.append(str(subList)[1:-1])  # omit brackets
        indexStart = lastIndex
    mergedList = ",".join(listStrings)
    expressionDecoded = (
        "numpy.array([" + mergedList + "],dtype=numpy."
        + arrayDtypeName
        + ").reshape"
        + str(arrayTarget.shape)
    )
    return f"{identifierName} = {expressionDecoded}"

def Z06convertNDArrayToStr(arrayTarget: numpy.ndarray, identifierName: str) -> str:
    # A binary modeling approach for demonstration; not actually compressing
    arrayFlattened = arrayTarget.ravel()
    asText = ",".join(str(val) for val in arrayFlattened)
    expressionDecoded = f"numpy.fromstring('{asText}',sep=',').reshape{arrayTarget.shape}"
    return f"{identifierName} = {expressionDecoded}"

def Z08convertNDArrayToStr(arrayTarget: numpy.ndarray, identifierName: str) -> str:
    #worse
    def rle(arrayInput):
        arrayLength = len(arrayInput)
        # if arrayLength == 0:
        #     return (None, None, None)
        arrayDiff = arrayInput[1:] != arrayInput[:-1]
        arrayIndices = numpy.append(numpy.where(arrayDiff), arrayLength - 1)
        arrayRunLengths = numpy.diff(numpy.append(-1, arrayIndices))
        arrayPositions = numpy.cumsum(numpy.append(0, arrayRunLengths))[:-1]
        arrayValues = arrayInput[arrayIndices]
        return (arrayRunLengths, arrayPositions, arrayValues)
    arrayFlattened = arrayTarget.ravel()
    arrayRunLengths, _1, arrayValues = rle(arrayFlattened)

    nestedRunLengths, _1, nestedValuesToRepeat = rle(arrayValues)
    targetShape = arrayValues.shape

    valuesToRepeat = f"R({nestedValuesToRepeat.tolist()},{nestedRunLengths.tolist()}).reshape{targetShape}"

    arrayAsTypeStr = f"R({valuesToRepeat}.tolist(),{arrayRunLengths.tolist()}).reshape{arrayTarget.shape}"

    return f"{identifierName} = numpy.array({arrayAsTypeStr},dtype=numpy.{arrayTarget.dtype})"

def Z09convertNDArrayToStr(arrayTarget: numpy.ndarray, identifierName: str) -> str:
    arrayAsTypeStr = numpy.array2string(arrayTarget, threshold=1000000, max_line_width=200, separator=',')
    stringMinimized = python_minifier.minify(arrayAsTypeStr)
    commaZeroMaximum = arrayTarget.shape[-1] - 1
    stringMinimized = stringMinimized.replace('[0' + ',0'*commaZeroMaximum + ']', '[0]*'+str(commaZeroMaximum+1))
    for countZeros in range(commaZeroMaximum, 2, -1):
        stringMinimized = stringMinimized.replace(',0'*countZeros + ']', ']+[0]*'+str(countZeros))
    return f"{identifierName} = numpy.array({stringMinimized}, dtype=numpy.{arrayTarget.dtype})"

def convertNDArrayToStr(arrayTarget: numpy.ndarray, identifierName: str) -> str:
    arrayAsTypeStr = numpy.array2string(arrayTarget, threshold=1000000, max_line_width=200, separator=',')
    stringMinimized = python_minifier.minify(arrayAsTypeStr)

    # Handle sequences of zeros first
    commaZeroMaximum = arrayTarget.shape[-1] - 1
    stringMinimized = stringMinimized.replace('[0' + ',0'*commaZeroMaximum + ']', '[0]*'+str(commaZeroMaximum+1))
    for countZeros in range(commaZeroMaximum, 2, -1):
        stringMinimized = stringMinimized.replace(',0'*countZeros + ']', ']+[0]*'+str(countZeros))

    # Find and replace incrementing sequences
    import re
    def replaceIncrementingSequence(match):
        sequence = [int(x) for x in match.group(0).split(',')]
        if len(sequence) < 3:  # Only replace if sequence is long enough
            return match.group(0)

        # Check if sequence is incrementing by 1
        if all(sequence[i+1] - sequence[i] == 1 for i in range(len(sequence)-1)):
            rangeStr = f'numpy.arange({sequence[0]},{sequence[-1]+1})'
            return rangeStr if len(rangeStr) < len(match.group(0)) else match.group(0)
        return match.group(0)

    # Find sequences of numbers separated by commas
    pattern = r'\d+(?:,\d+)+'
    stringMinimized = re.sub(pattern, replaceIncrementingSequence, stringMinimized)

    return f"{identifierName} = numpy.array({stringMinimized}, dtype=numpy.{arrayTarget.dtype})"
COUNTfunctions = 10

for id in range(COUNTfunctions):
    idAsStr = f'Z{str(id).zfill(2)}'
    callableComparand = globals()[f'{idAsStr}convertNDArrayToStr']

    stringReturned = python_minifier.minify(callableComparand(connectionGraph, idAsStr))
    # print(stringReturned)
    exec(stringReturned)
    assert numpy.array_equal(connectionGraph, globals()[idAsStr])
    dictionaryLengths[idAsStr] = len(stringReturned)

for idAsStr, lengthString in sorted(dictionaryLengths.items(), key=lambda item: item[1]):
    print(f"{idAsStr}: {lengthString}")

"""
def Z03convertNDArrayToStr(arrayTarget: numpy.ndarray, identifierName: str) -> str:
    # doesn't work
    #Compress using [x]*N for repeats and range/arange for sequences
    def optimizeSequence(arraySegment):
        #Choose best representation between direct, multiply, or range
        if len(arraySegment) < 3:
            return str(arraySegment.tolist())

        if numpy.all(arraySegment == arraySegment[0]):
            strMultiply = f"[{arraySegment[0]}]*{len(arraySegment)}"
            strDirect = str(arraySegment.tolist())
            return strMultiply if len(strMultiply) < len(strDirect) else strDirect

        arrayDiff = numpy.diff(arraySegment)
        if len(arraySegment) >= 4 and numpy.all(arrayDiff == arrayDiff[0]):
            step = int(arrayDiff[0])
            strRange = f"list(A({arraySegment[0]},{arraySegment[-1]+step},{step}))"
            strDirect = str(arraySegment.tolist())
            return strRange if len(strRange) < len(strDirect) else strDirect

        return str(arraySegment.tolist())

    if arrayTarget.size == 0:
        arrayAsTypeStr = "numpy.array([])"
    else:
        listCompressed = []
        for index in numpy.ndindex(arrayTarget.shape[:-1]):
            arrayRow = arrayTarget[index]
            strCompressed = optimizeSequence(arrayRow)
            listCompressed.append(strCompressed)

        arrayAsTypeStr = f"numpy.array([{','.join(listCompressed)}])"

    return f"{identifierName} = numpy.array({arrayAsTypeStr}, dtype=numpy.{arrayTarget.dtype})"
def Z04convertNDArrayToStr(arrayTarget: numpy.ndarray, identifierName: str) -> str:
    # doesn't work
    def rle(arrayInput):
        arrayLength = len(arrayInput)
        arrayDiff = arrayInput[1:] != arrayInput[:-1]
        arrayIndices = numpy.append(numpy.where(arrayDiff), arrayLength - 1)
        arrayRunLengths = numpy.diff(numpy.append(-1, arrayIndices))
        arrayValues = arrayInput[arrayIndices]
        return arrayRunLengths, arrayValues

    def optimizeSequence(value, count):
        strMultiply = f"[{value}]*{count}"
        strNormal = str(value)
        return strMultiply if len(strMultiply) < len(strNormal) * count else value

    arrayFlattened = arrayTarget.ravel()
    arrayRunLengths, arrayValues = rle(arrayFlattened)

    listOptimized = []
    for value, count in zip(arrayValues, arrayRunLengths):
        listOptimized.append(optimizeSequence(value, count))

    arrayAsTypeStr = f"numpy.hstack([{','.join(map(str, listOptimized))}]).reshape{arrayTarget.shape}"
    return f"{identifierName} = numpy.array({arrayAsTypeStr},dtype=numpy.{arrayTarget.dtype})"
def Z09convertNDArrayToStr(arrayTarget: numpy.ndarray, identifierName: str) -> str:
    # doesn't work
    def rle(arrayInput):
        arrayLength = len(arrayInput)
        arrayDiff = numpy.concatenate(([True], arrayInput[1:] != arrayInput[:-1], [True]))
        arrayIndices = numpy.nonzero(arrayDiff)[0]
        arrayRunLengths = numpy.diff(arrayIndices)
        arrayValues = arrayInput[arrayIndices[:-1]]
        return arrayRunLengths, arrayValues

    def optimizeLongRuns(values, lengths):
        COUNTminCompression = 4  # Only compress runs longer than this
        strParts = []
        for v, n in zip(values, lengths):
            if n > COUNTminCompression and len(f"[{v}]*{n}") < len(str(v)) * n:
                strParts.append(f"[{v}]*{n}")
            else:
                strParts.extend([str(v)] * n)
        return f"numpy.array([{','.join(strParts)}])"

    arrayRunLengths, arrayValues = rle(arrayTarget.ravel())
    arrayAsTypeStr = f"{optimizeLongRuns(arrayValues, arrayRunLengths)}.reshape{arrayTarget.shape}"
    return f"{identifierName} = numpy.array({arrayAsTypeStr},dtype=numpy.{arrayTarget.dtype})"


"""
