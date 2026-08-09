# The real module is generated from this incomplete module. Comments are not preserved.
# ruff: file-ignore[global-statement, typing-only-first-party-import]
# ty: ignore[invalid-argument-type]
from __future__ import annotations

from copy import deepcopy
from mapFolding import DatatypeFoldsTotal
from mapFolding.dataBaskets import SymmetricFoldsState
from queue import Queue
from threading import Lock, Thread

boxOfThreads: list[Thread] = []
queueFutures: Queue[SymmetricFoldsState] = Queue()
symmetricFoldsTotal: int = 0
LOCKsymmetricFoldsTotal = Lock()
# TODO There isn't a better way to do this?
STOPsignal = object()
# pyright: reportArgumentType=false

def initializeConcurrencyManager(maxWorkers: int, symmetricFolds: int = 0) -> None:
	global boxOfThreads, symmetricFoldsTotal, queueFutures
	boxOfThreads = []
	queueFutures = Queue()
	symmetricFoldsTotal = symmetricFolds

	indexThread = 0
	while indexThread < maxWorkers:
		thread = Thread(target=_threadDoesSomething, name=f"thread{indexThread}", daemon=True)
		thread.start()
		boxOfThreads.append(thread)
		indexThread += 1

def _threadDoesSomething() -> None:
	global symmetricFoldsTotal
	while True:
		state: SymmetricFoldsState = queueFutures.get()
		if state is STOPsignal:
			break
		state = _filterAsymmetricFolds(state)
		with LOCKsymmetricFoldsTotal:
			symmetricFoldsTotal += state.symmetricFolds

def _filterAsymmetricFolds(state: SymmetricFoldsState) -> SymmetricFoldsState:
	"""Add real function during generation; the signature is here to preview its interactions with the module."""
	return state

def filterAsymmetricFolds(state: SymmetricFoldsState) -> None:
	queueFutures.put_nowait(deepcopy(state))

def getSymmetricFoldsTotal() -> DatatypeFoldsTotal:
	for _thread in boxOfThreads:
		queueFutures.put(STOPsignal)
	for thread in boxOfThreads:
		thread.join()
	return symmetricFoldsTotal
