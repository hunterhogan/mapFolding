from __future__ import annotations

from mapFolding.algorithms.permutationsBilateral import (
	_updateMilepost, crossRoadDiagonally, Interval, makeMilepost, Milepost, ToDo, uncrossRoadLaterally, Undo, whatToDo)
from multiprocessing import get_context, Queue
from threading import Thread
from typing import TYPE_CHECKING
import copy
import os

if TYPE_CHECKING:
	from multiprocessing.context import DefaultContext
	from typing import Any

def _countTask(qq: tuple[int, Interval, Milepost, int]) -> int:
	n下i, interval, milepost, n = qq
	totalPermutations: int = 0
	toDoUndoLIFO: list[tuple[int, Interval, Milepost] | Undo] = [(n下i, interval, milepost)]

	while toDoUndoLIFO:
		work: tuple[int, Interval, Milepost] | Undo = toDoUndoLIFO.pop()
		if len(work) == 3:
			n下i, interval, milepost = work  # pyright: ignore[reportAssignmentType]

			if interval is not milepost.interval:
				interval, milepost, *undo = crossRoadDiagonally(interval, milepost)
				toDoUndoLIFO.append(Undo(n下i, interval, milepost, *undo))

				是: ToDo = whatToDo(n下i, n)
				if 是.keepCounting:
					if 是.countComplement:
						toDoUndoLIFO.append((是.n下i, interval.milepost.complement.intervalComplement.distal, interval.milepost.complement))
					toDoUndoLIFO.append((是.n下i, interval.milepost.intervalComplement.distal, interval.milepost))
				else:
					totalPermutations += 1
		else:
			toDoUndoLIFO.append(uncrossRoadLaterally(work))
	return totalPermutations

def _workerLoop(queueIn: Queue[Any], queueOut: Queue[Any]) -> None:
	while True:
		task = queueIn.get()
		if task is None:
			break
		queueOut.put(_countTask(task))

def _gatherFrontier(n下i: int, milepost: Milepost, n: int, splitDepth: int, *, countComplement: bool) -> tuple[list[tuple[int, Interval, Milepost, int]], int]:
	"""Run DFS to a fixed recursion depth, deep copying each frontier node as an independent task.

	Splitting at n//2 levels produces many small tasks of similar size so the process
	pool can drain them with natural work-stealing. Leaves (keepCounting=False) are counted
	directly because the original count() increments by 1 there and does not recurse further.
	"""
	listQQ: list[tuple[int, Interval, Milepost, int]] = []
	totalFU: int = 0
	# splitN: children whose n下i reaches this value are snapshotted, not expanded
	jjjjjjjjjjjjjjjjjjjjjjjjjjjj: int = n下i + splitDepth
	toDoUndoLIFO: list[tuple[int, Interval, Milepost] | Undo] = [(n下i, milepost.intervalComplement.distal, milepost)]
	if countComplement:
		toDoUndoLIFO.append((n下i, milepost.complement.intervalComplement.distal, milepost.complement))

	while toDoUndoLIFO:
		work: tuple[int, Interval, Milepost] | Undo = toDoUndoLIFO.pop()
		if len(work) != 3:
			toDoUndoLIFO.append(uncrossRoadLaterally(work))
			continue

		n下i, interval, milepost = work  # pyright: ignore[reportAssignmentType]

		if interval is not milepost.interval:
			interval, milepost, *undo = crossRoadDiagonally(interval, milepost)
			undoItem: Undo = Undo(n下i, interval, milepost, *undo)

			是: ToDo = whatToDo(n下i, n)

			if not 是.keepCounting:
				# leaf: original count() increments by 1 here and does NOT recurse further
				totalFU += 1
				toDoUndoLIFO.append(undoItem)
			elif jjjjjjjjjjjjjjjjjjjjjjjjjjjj <= 是.n下i:
				# children are at the split depth: deepcopy each as an independent task
				listQQ.append(copy.deepcopy((是.n下i, interval.milepost.intervalComplement.distal, interval.milepost, n)))
				if 是.countComplement:
					listQQ.append(copy.deepcopy((是.n下i, interval.milepost.complement.intervalComplement.distal, interval.milepost.complement, n)))
				toDoUndoLIFO.append(undoItem)
			else:
				# keep expanding; undoItem pops last, after both children complete
				toDoUndoLIFO.append(undoItem)
				if 是.countComplement:
					toDoUndoLIFO.append((是.n下i, interval.milepost.complement.intervalComplement.distal, interval.milepost.complement))
				toDoUndoLIFO.append((是.n下i, interval.milepost.intervalComplement.distal, interval.milepost))

	return listQQ, totalFU

def doTheNeedful(n: int, *, symmetric: bool) -> int:
	totalPermutations: int = 1
	milepost: Milepost = makeMilepost()
	n下i: int = 1

	是: ToDo = whatToDo(n下i, n)
	if 是.keepCounting:
		_updateMilepost(milepost, makeMilepost())
		_updateMilepost(milepost.complement, makeMilepost().complement)

	是 = whatToDo(n下i, n)
	if not 是.keepCounting:
		return totalPermutations

	cpuCount: int = os.cpu_count() or 1
	# Split halfway down the recursion tree: produces O(3^(n/2)) tasks, each small enough
	# that the pool drains them with negligible imbalance across all available cores.
	splitDepth: int = max(4, n // 2)

	tasks, directLeafCount = _gatherFrontier(是.n下i, milepost, n, splitDepth, countComplement=是.countComplement)

	processContext: DefaultContext = get_context()
	queueIn: Queue[Any] = processContext.Queue(maxsize=cpuCount * 4)
	queueOut: Queue[Any] = processContext.Queue()

	boxOfProcesses = [processContext.Process(target=_workerLoop, args=(queueIn, queueOut)) for _cpu in range(cpuCount)]
	for process in boxOfProcesses:
		process.start()

	def howManyLayersOfFunctionsAreThere() -> None:
		for task in tasks:
			queueIn.put(task)
		for _cpu in range(cpuCount):
			queueIn.put(None)

	feederIsAStupidName: Thread = Thread(target=howManyLayersOfFunctionsAreThere)
	feederIsAStupidName.start()

	totalPermutations = directLeafCount + sum(queueOut.get() for _task in range(len(tasks)))

	feederIsAStupidName.join()
	for process in boxOfProcesses:
		process.join()
	queueIn.close()
	queueOut.close()

	return totalPermutations * (2 - symmetric)
