# The real module is generated from this incomplete module. Comments are not preserved.
# ruff: file-ignore[global-statement, typing-only-first-party-import]
# ty: ignore[invalid-argument-type]
from __future__ import annotations

from copy import deepcopy
from mapFolding.dataBaskets import SymmetricFoldsState
from mapFolding.theTypes import 形TotalFolds
from queue import Queue
from threading import Lock, Thread

boxOfThreads: list[Thread] = []
queueFutures: Queue[SymmetricFoldsState] = Queue()
symmetricTotalFolds: int = 0
LOCKsymmetricTotalFolds = Lock()
# TODO There isn't a better way to do this?
STOPsignal = object()
# pyright: reportArgumentType=false

def initializeConcurrencyManager(maxWorkers: int, symmetricFolds: int = 0) -> None:
	global boxOfThreads, symmetricTotalFolds, queueFutures
	boxOfThreads = []
	queueFutures = Queue()
	symmetricTotalFolds = symmetricFolds

	indexThread = 0
	while indexThread < maxWorkers:
		thread = Thread(target=_threadDoesSomething, name=f"thread{indexThread}", daemon=True)
		thread.start()
		boxOfThreads.append(thread)
		indexThread += 1

def _threadDoesSomething() -> None:
	global symmetricTotalFolds
	while True:
		state: SymmetricFoldsState = queueFutures.get()
		if state is STOPsignal:
			break
		state = _filterAsymmetricFolds(state)
		with LOCKsymmetricTotalFolds:
			symmetricTotalFolds += state.symmetricFolds

def _filterAsymmetricFolds(state: SymmetricFoldsState) -> SymmetricFoldsState:
	"""Add real function during generation; the signature is here to preview its interactions with the module."""
	return state

def filterAsymmetricFolds(state: SymmetricFoldsState) -> None:
	queueFutures.put_nowait(deepcopy(state))

def getSymmetricTotalFolds() -> 形TotalFolds:
	for _thread in boxOfThreads:
		queueFutures.put(STOPsignal)
	for thread in boxOfThreads:
		thread.join()
	return symmetricTotalFolds
