"""Count semi-meanders with independent bilateral search tasks in worker processes.

(AI generated docstring)

You can divide a prepared bilateral search into independent tasks and combine their counts
for ordinary or symmetric semi-meanders.

Contents
--------
Functions
	doTheNeedful
		Prepare search tasks, collect worker counts, and apply the family multiplier.
"""
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
	"""I use this task to count completions of an independent bilateral search branch.

	(AI generated docstring)

	Input Mutation
	--------------
	The task's linked interval structure changes as crossings are added and undone.

	Parameters
	----------
	qq : tuple[int, Interval, Milepost, int]
		Crossing number, first interval to explore, prepared milepost, and final crossing number.

	Returns
	-------
	totalPermutations : int
		Number of completed branches in the task, before the family multiplier.
	"""
	n下i, interval, milepost, n = qq
	totalPermutations: int = 0
	toDoUndoLIFO: list[tuple[int, Interval, Milepost] | Undo] = [(n下i, interval, milepost)]

	while toDoUndoLIFO:
		work: tuple[int, Interval, Milepost] | Undo = toDoUndoLIFO.pop()
		if len(work) == 3:
			n下i, interval, milepost = work

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
	"""I use this worker to return task counts until the input queue supplies `None`.

	(AI generated docstring)

	Queue Consumption
	-----------------
	Each search task is removed from `queueIn`, and its completed count is put on `queueOut`.
	A `None` item ends the worker without producing a result for that item.

	Parameters
	----------
	queueIn : Queue[Any]
		Queue supplying bilateral search tasks or the stopping value `None`.
	queueOut : Queue[Any]
		Queue receiving one integer count for each completed task.
	"""
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

		n下i, interval, milepost = work

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

def doTheNeedful(n: int, *, symmetric: bool, concurrencyLimit: int | None = None) -> int:
	"""Count ordinary or symmetric semi-meanders across worker processes.

	(AI generated docstring)

	Worker Processes
	----------------
	The search creates independent tasks, starts the selected number of processes, and waits
	for every task result and process before returning the combined count.

	Parameters
	----------
	n : int
		For ordinary semi-meanders, one less than the number of crossings. For symmetric
		semi-meanders, the number of crossings.
	symmetric : bool
		`True` selects symmetric semi-meanders at index `n`. `False` selects ordinary
		semi-meanders at index `n + 1` and doubles the representative count when `n` exceeds one.
	concurrencyLimit : int | None = None
		Positive worker count. `None` uses the reported CPU count, or one when unavailable.

	Returns
	-------
	totalPermutations : int
		Combined count for the selected family. Indices at most one return one without workers.
	"""
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

	cpuCount: int = (os.cpu_count() or 1) if concurrencyLimit is None else concurrencyLimit
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
		"""Supply every search task and one stopping value per worker to the input queue.

		(AI generated docstring)

		Queue Population
		----------------
		The enclosing `queueIn` receives the prepared tasks followed by `cpuCount` values
		of `None`. The feeder waits when the queue is full so workers can consume the tasks.
		"""
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
