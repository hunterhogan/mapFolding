from mapFolding import packageSettings
from mapFolding._e import getLeavesCreasePost
from mapFolding._e.dataBaskets import EliminationState
from pathlib import Path, PurePath
from typing import TextIO

def subdivideP2d7s0_1_3_2CSVFile(state: EliminationState, pathDataRaw: Path) -> None:
	pathSorted: Path = pathDataRaw / "sorted"
	pathSorted.mkdir(exist_ok=True)

	pathFilenameSource: Path = pathDataRaw / "p2d7s0_1_3_2.csv"
	if pathFilenameSource.exists():
		setLeavesAllowedAfterTwo: set[int] = set(getLeavesCreasePost(state, 2))

		dictionaryAppendStreams: dict[int, TextIO] = {}
		try:
			with pathFilenameSource.open('r', newline='') as readStream:
				for lineRaw in readStream:
					line: str = lineRaw.rstrip('\n').rstrip('\r')
					if len(line) != 401:
						continue
					if line.count(',') != 127:
						continue
					if not line.startswith("0,1,3,2,"):
						continue
					if line[0] == ',' or line[-1] == ',' or ',,' in line:
						continue

					listPrefixParts: list[str] = line.split(',', 5)
					if len(listPrefixParts) < 6:
						continue
					if not listPrefixParts[4].isdigit():
						continue
					leafFifth: int = int(listPrefixParts[4])
					if leafFifth not in setLeavesAllowedAfterTwo:
						continue

					appendStream: TextIO | None = dictionaryAppendStreams.get(leafFifth)
					if appendStream is None:
						pathFilenameOutput: Path = pathDataRaw / f"p2d7s0_1_3_2_{leafFifth}.csv"
						appendStream = pathFilenameOutput.open('a', newline='')
						dictionaryAppendStreams[leafFifth] = appendStream

					appendStream.write(line)
					appendStream.write('\n')

			pathFilenameDestination: Path = pathSorted / pathFilenameSource.name
			pathFilenameSource.replace(pathFilenameDestination)
		finally:
			for appendStream in dictionaryAppendStreams.values():
				appendStream.close()

def cleanAndSortSequencesCSVFile(state: EliminationState, pathFilename: PurePath) -> None:
	pathFilenameTarget: Path = Path(pathFilename)
	pathSorted: Path = pathFilenameTarget.parent / "sorted"
	pathSorted.mkdir(exist_ok=True)

	lineHeader: str | None = None
	tupleHeaderExpected: tuple[int, ...] = tuple(range(state.leavesTotal))

	setSequences: set[tuple[int, ...]] = set()
	listSequencesUnique: list[tuple[int, ...]] = []

	duplicatesDetected: bool = False
	invalidLinesDetected: bool = False
	sortedAlready: bool = True
	sequencePrior: tuple[int, ...] | None = None

	with pathFilenameTarget.open('r', newline='') as readStream:
		for indexLine, lineRaw in enumerate(readStream):
			line: str = lineRaw.rstrip('\n').rstrip('\r')
			if indexLine == 0 and line.startswith("0,1,2,"):
				listHeaderParts: list[str] = line.split(',')
				if len(listHeaderParts) == state.leavesTotal:
					try:
						tupleHeaderFound: tuple[int, ...] = tuple(int(part) for part in listHeaderParts)
					except ValueError:
						tupleHeaderFound = ()
					if tupleHeaderFound == tupleHeaderExpected:
						lineHeader = line
						continue

			if not line:
				continue
			if line[0] == ',' or line[-1] == ',' or ',,' in line:
				invalidLinesDetected = True
				continue
			if line.count(',') != state.leavesTotal - 1:
				invalidLinesDetected = True
				continue
			try:
				tupleSequence: tuple[int, ...] = tuple(int(part) for part in line.split(','))
			except ValueError:
				invalidLinesDetected = True
				continue
			if len(tupleSequence) != state.leavesTotal:
				invalidLinesDetected = True
				continue

			if sequencePrior is not None and tupleSequence < sequencePrior:
				sortedAlready = False
			sequencePrior = tupleSequence

			if tupleSequence in setSequences:
				duplicatesDetected = True
				continue
			setSequences.add(tupleSequence)
			listSequencesUnique.append(tupleSequence)

	if not (duplicatesDetected or invalidLinesDetected or not sortedAlready):
		return

	listSequencesSorted: list[tuple[int, ...]] = sorted(listSequencesUnique)
	pathFilenameBackup: Path = pathSorted / pathFilenameTarget.name
	pathFilenameTarget.replace(pathFilenameBackup)
	with pathFilenameTarget.open('w', newline='') as writeStream:
		if lineHeader is not None:
			writeStream.write(lineHeader)
			writeStream.write('\n')
		for tupleSequence in listSequencesSorted:
			writeStream.write(','.join(str(value) for value in tupleSequence))
			writeStream.write('\n')

def sortP2d7GeneratedCSVFiles(state: EliminationState, pathDataRaw: Path) -> None:
	pathSorted: Path = pathDataRaw / "sorted"
	pathSorted.mkdir(exist_ok=True)

	setLeavesAllowedAfterOne: set[int] = set(getLeavesCreasePost(state, 1))
	dictionaryAllowedAfterThird: dict[int, set[int]] = {
		leafThird: set(getLeavesCreasePost(state, leafThird))
		for leafThird in setLeavesAllowedAfterOne
	}

	dictionaryAppendStreams: dict[tuple[int, int], TextIO] = {}
	try:
		for pathFilenameSource in sorted(pathDataRaw.glob("p2d7_*.csv")):
			with pathFilenameSource.open('r', newline='') as readStream:
				for lineRaw in readStream:
					line: str = lineRaw.rstrip('\n').rstrip('\r')
					if len(line) != 401:
						continue
					if line.count(',') != 127:
						continue
					if not line.startswith("0,1,"):
						continue
					if line[0] == ',' or line[-1] == ',' or ',,' in line:
						continue

					listPrefixParts: list[str] = line.split(',', 4)
					if len(listPrefixParts) < 5:
						continue
					if not listPrefixParts[2].isdigit() or not listPrefixParts[3].isdigit():
						continue
					leafThird: int = int(listPrefixParts[2])
					leafFourth: int = int(listPrefixParts[3])
					if leafThird not in setLeavesAllowedAfterOne:
						continue
					if leafFourth not in dictionaryAllowedAfterThird[leafThird]:
						continue

					key: tuple[int, int] = (leafThird, leafFourth)
					appendStream: TextIO | None = dictionaryAppendStreams.get(key)
					if appendStream is None:
						pathFilenameOutput: Path = pathDataRaw / f"p2d7s0_1_{leafThird}_{leafFourth}.csv"
						appendStream = pathFilenameOutput.open('a', newline='')
						dictionaryAppendStreams[key] = appendStream

					appendStream.write(line)
					appendStream.write('\n')

			pathFilenameDestination: Path = pathSorted / pathFilenameSource.name
			pathFilenameSource.replace(pathFilenameDestination)
	finally:
		for appendStream in dictionaryAppendStreams.values():
			appendStream.close()

if __name__ == '__main__':
	sortEm = True
	if sortEm:
		state = EliminationState((2,) * 7)
		pathDataRaw: Path = packageSettings.pathPackage / "_e" / "dataRaw"
		sortP2d7GeneratedCSVFiles(state, pathDataRaw)
		subdivideP2d7s0_1_3_2CSVFile(state, pathDataRaw)
		for pathFilename in pathDataRaw.glob("p2d7s*.csv"):
			cleanAndSortSequencesCSVFile(state, pathFilename)

	# type \apps\mapFolding\mapFolding\_e\dataRaw\p2d7s*.csv | find /c /v ""
	# 521292 of 562368 😢
