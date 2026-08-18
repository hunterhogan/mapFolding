from __future__ import annotations

from mapFolding._e._2上nDimensional import getLeavesCreasePost
from mapFolding._e.dataBaskets import StateElimination
from mapFolding.theSSOT import settingsPackage
from pathlib import Path, PurePath
from typing import TextIO

def subdivideP2d7s0_1_3_2CSVFile(state: StateElimination, pathDataRaw: Path) -> None:
	pathSorted: Path = pathDataRaw / "sorted"
	pathSorted.mkdir(exist_ok=True)

	pathFilenameSource: Path = pathDataRaw / "p2d7s0_1_3_2.csv"
	if pathFilenameSource.exists():
		boxOfLeavesAllowedAfterTwo: set[int] = set(getLeavesCreasePost(state, 2))

		dictionaryAppendStreams: dict[int, TextIO] = {}
		try:
			with pathFilenameSource.open('r', encoding="utf-8", newline='') as readStream:
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

					boxOfPrefixParts: list[str] = line.split(',', 5)
					if len(boxOfPrefixParts) < 6:
						continue
					if not boxOfPrefixParts[4].isdigit():
						continue
					leafFifth: int = int(boxOfPrefixParts[4])
					if leafFifth not in boxOfLeavesAllowedAfterTwo:
						continue

					appendStream: TextIO | None = dictionaryAppendStreams.get(leafFifth)
					if appendStream is None:
						pathFilenameOutput: Path = pathDataRaw / f"p2d7s0_1_3_2_{leafFifth}.csv"
						appendStream = pathFilenameOutput.open('a', encoding="utf-8", newline='')
						dictionaryAppendStreams[leafFifth] = appendStream

					appendStream.write(line)
					appendStream.write('\n')

			pathFilenameDestination: Path = pathSorted / pathFilenameSource.name
			pathFilenameSource.replace(pathFilenameDestination)
		finally:
			for appendStream in dictionaryAppendStreams.values():
				appendStream.close()

def cleanAndSortSequencesCSVFile(state: StateElimination, pathFilename: PurePath) -> None:
	pathFilenameTarget: Path = Path(pathFilename)
	pathSorted: Path = pathFilenameTarget.parent / "sorted"
	pathSorted.mkdir(exist_ok=True)

	lineHeader: str | None = None
	boxOfHeaderExpected: tuple[int, ...] = tuple(range(state.totalLeaves))

	boxOfSequences: set[tuple[int, ...]] = set()
	boxOfSequencesUnique: list[tuple[int, ...]] = []

	duplicatesDetected: bool = False
	invalidLinesDetected: bool = False
	sortedAlready: bool = True
	sequencePrior: tuple[int, ...] | None = None

	with pathFilenameTarget.open('r', encoding="utf-8", newline='') as readStream:
		for 次Line, lineRaw in enumerate(readStream):
			line: str = lineRaw.rstrip('\n').rstrip('\r')
			if 次Line == 0 and line.startswith("0,1,2,"):
				boxOfHeaderParts: list[str] = line.split(',')
				if len(boxOfHeaderParts) == state.totalLeaves:
					try:
						boxOfHeaderFound: tuple[int, ...] = tuple(int(part) for part in boxOfHeaderParts)
					except ValueError:
						boxOfHeaderFound = ()
					if boxOfHeaderFound == boxOfHeaderExpected:
						lineHeader = line
						continue

			if not line:
				continue
			if line[0] == ',' or line[-1] == ',' or ',,' in line:
				invalidLinesDetected = True
				continue
			if line.count(',') != state.totalLeaves - 1:
				invalidLinesDetected = True
				continue
			try:
				boxOfSequence: tuple[int, ...] = tuple(int(part) for part in line.split(','))
			except ValueError:
				invalidLinesDetected = True
				continue
			if len(boxOfSequence) != state.totalLeaves:
				invalidLinesDetected = True
				continue

			if sequencePrior is not None and boxOfSequence < sequencePrior:
				sortedAlready = False
			sequencePrior = boxOfSequence

			if boxOfSequence in boxOfSequences:
				duplicatesDetected = True
				continue
			boxOfSequences.add(boxOfSequence)
			boxOfSequencesUnique.append(boxOfSequence)

	if not (duplicatesDetected or invalidLinesDetected or not sortedAlready):
		return

	boxOfSequencesSorted: list[tuple[int, ...]] = sorted(boxOfSequencesUnique)
	pathFilenameBackup: Path = pathSorted / pathFilenameTarget.name
	pathFilenameTarget.replace(pathFilenameBackup)
	with pathFilenameTarget.open('w', encoding="utf-8", newline='') as writeStream:
		if lineHeader is not None:
			writeStream.write(lineHeader)
			writeStream.write('\n')
		for boxOfSequence in boxOfSequencesSorted:
			writeStream.write(','.join(str(value) for value in boxOfSequence))
			writeStream.write('\n')

def sortP2d7GeneratedCSVFiles(state: StateElimination, pathDataRaw: Path) -> None:
	pathSorted: Path = pathDataRaw / "sorted"
	pathSorted.mkdir(exist_ok=True)

	boxOfLeavesAllowedAfterOne: set[int] = set(getLeavesCreasePost(state, 1))
	dictionaryAllowedAfterThird: dict[int, set[int]] = {
		leafThird: set(getLeavesCreasePost(state, leafThird))
		for leafThird in boxOfLeavesAllowedAfterOne
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

					boxOfPrefixParts: list[str] = line.split(',', 4)
					if len(boxOfPrefixParts) < 5:
						continue
					if not boxOfPrefixParts[2].isdigit() or not boxOfPrefixParts[3].isdigit():
						continue
					leafThird: int = int(boxOfPrefixParts[2])
					leafFourth: int = int(boxOfPrefixParts[3])
					if leafThird not in boxOfLeavesAllowedAfterOne:
						continue
					if leafFourth not in dictionaryAllowedAfterThird[leafThird]:
						continue

					key: tuple[int, int] = (leafThird, leafFourth)
					appendStream: TextIO | None = dictionaryAppendStreams.get(key)
					if appendStream is None:
						pathFilenameOutput: Path = pathDataRaw / f"p2d7s0_1_{leafThird}_{leafFourth}.csv"
						appendStream = pathFilenameOutput.open('a', encoding="utf-8", newline='')
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
		state = StateElimination((2,) * 7)
		pathDataRaw: Path = settingsPackage.pathPackage / "_e" / '_development' / "dataRaw"
		sortP2d7GeneratedCSVFiles(state, pathDataRaw)
		subdivideP2d7s0_1_3_2CSVFile(state, pathDataRaw)
		for pathFilename in pathDataRaw.glob("p2d7s*.csv"):
			cleanAndSortSequencesCSVFile(state, pathFilename)

	# type \apps\mapFolding\mapFolding\_e\_development\dataRaw\p2d7s*.csv | find /c /v ""
	# 521292 of 562368 😢
	# 523486
