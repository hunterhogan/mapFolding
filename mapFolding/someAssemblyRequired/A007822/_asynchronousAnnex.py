# NOTE The real module is generated from this incomplete module. Comments are not preserved.
# ruff: noqa: PLW0603
from __future__ import annotations

from copy import deepcopy
from queue import Queue
from threading import Lock, Thread
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding import DatatypeFoldsTotal
	from mapFolding.dataBaskets import SymmetricFoldsState

listThreads: list[Thread] = []
queueFutures: Queue[SymmetricFoldsState] = Queue()
symmetricFoldsTotal: int = 0
LOCKsymmetricFoldsTotal = Lock()
# TODO There isn't a better way to do this?
STOPsignal = object()

def initializeConcurrencyManager(maxWorkers: int, symmetricFolds: int = 0) -> None:
	global listThreads, symmetricFoldsTotal, queueFutures
	listThreads = []
	queueFutures = Queue()
	symmetricFoldsTotal = symmetricFolds

	indexThread = 0
	while indexThread < maxWorkers:
		thread = Thread(target=_threadDoesSomething, name=f"thread{indexThread}", daemon=True)
		thread.start()
		listThreads.append(thread)
		indexThread += 1

def _threadDoesSomething() -> None:
	global symmetricFoldsTotal
	while True:
		state = queueFutures.get()
		if state is STOPsignal:
			break
		state = _filterAsymmetricFolds(state)
		with LOCKsymmetricFoldsTotal:
			symmetricFoldsTotal += state.symmetricFolds

def _filterAsymmetricFolds(state: SymmetricFoldsState) -> SymmetricFoldsState:
	"""Add real function during generation; the signature is here to preview its interactions with the module."""  # noqa: DOC201
	return state

def filterAsymmetricFolds(state: SymmetricFoldsState) -> None:
	queueFutures.put_nowait(deepcopy(state))

def getSymmetricFoldsTotal() -> DatatypeFoldsTotal:
	for _thread in listThreads:
		queueFutures.put(STOPsignal)  # pyright: ignore[reportArgumentType]  # ty:ignore[invalid-argument-type]
	for thread in listThreads:
		thread.join()
	return symmetricFoldsTotal
