from mapFolding import indexMy, indexTrack, getAlgorithmSource
from someAssemblyRequired import makeInlineFunction
import pathlib
import inspect

algorithmSource = getAlgorithmSource()

def getDictionaryEnumValues():
    dictionaryEnumValues = {}
    for enumIndex in [indexMy, indexTrack]:
        for memberName, memberValue in enumIndex._member_map_.items():
            dictionaryEnumValues[f"{enumIndex.__name__}.{memberName}.value"] = memberValue.value
    return dictionaryEnumValues

def unpackArrays(codeInlined: str, callableTarget: str) -> str:
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

def inlineMapFoldingNumba():
    dictionaryEnumValues = getDictionaryEnumValues()
    codeSource = inspect.getsource(algorithmSource)
    pathFilenameAlgorithm = pathlib.Path(inspect.getfile(algorithmSource))

    listCallables = [ 'countInitialize', 'countParallel', 'countSequential', ]

    listPathFilenamesDestination: list[pathlib.Path] = []
    for callableTarget in listCallables:
        skipEnum = (callableTarget == 'countInitialize')
        skipEnum = (callableTarget == 'countSequential')
        pathFilenameDestination = pathFilenameAlgorithm.parent / "syntheticModules" / pathFilenameAlgorithm.with_stem(callableTarget).name
        codeInlined, callableInlinedDecorators, importsRequired = makeInlineFunction(codeSource, callableTarget, dictionaryEnumValues, skipEnum)
        codeUnpacked = unpackArrays(codeInlined, callableTarget)
        pathFilenameDestination.write_text(importsRequired + "\n" + codeUnpacked)
        listPathFilenamesDestination.append(pathFilenameDestination)

if __name__ == '__main__':
    inlineMapFoldingNumba()
