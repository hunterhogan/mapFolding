# The real module is generated from this incomplete module. Comments are not preserved. Docstrings corrupt the file.
# ruff: file-ignore[global-statement, typing-only-first-party-import]
# ty: ignore[invalid-argument-type]
from __future__ import annotations

from copy import deepcopy
from mapFolding.dataBaskets import StateMapFoldingSymmetric
from mapFolding.theTypes import 形TotalFolds
from queue import Queue
from threading import Lock, Thread

boxOfThreads: list[Thread] = []
queueFutures: Queue[StateMapFoldingSymmetric] = Queue()
symmetricTotalFolds: int = 0
LOCKsymmetricTotalFolds = Lock()
# TODO There isn't a better way to do this?
STOPsignal = object()
# pyright: reportArgumentType=false

def initializeConcurrencyManager(maxWorkers: int, symmetricFolds: int = 0) -> None:
	"""Initialize the shared queue, accumulated count, and background workers.

	(AI generated docstring)

	You can use this function in a generated module to reset shared worker state and start
	`maxWorkers` daemon threads. Each thread consumes the shared queue and accumulates filter
	results. The source template contains a placeholder filter; generation supplies the real filter.

	Parameters
	----------
	maxWorkers : int
		Number of threads to start; a nonpositive value starts no threads.
	symmetricFolds : int = 0
		Initial value for the shared accumulated symmetric count.

	Examples
	--------
	The module `mapFolding.synthesized.mapFoldingSymmetric.asynchronous` uses this call.

		```python
		initializeConcurrencyManager(maxWorkers)
		```
	"""
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
	"""I use this worker target to consume queued states and accumulate filtered counts.

	(AI generated docstring)

	This function waits for queued state, applies the generated filter, and adds the resulting
	symmetric count while holding the shared lock. The function exits after receiving the shared
	stop object.
	"""
	global symmetricTotalFolds
	while True:
		state: StateMapFoldingSymmetric = queueFutures.get()
		if state is STOPsignal:
			break
		state = _filterAsymmetricFolds(state)
		with LOCKsymmetricTotalFolds:
			symmetricTotalFolds += state.symmetricFolds

def _filterAsymmetricFolds(state: StateMapFoldingSymmetric) -> StateMapFoldingSymmetric:
	"""I use this placeholder to define the filtering contract in the worker template.

	(AI generated docstring)

	This function returns `state`. The asynchronous module generator replaces this definition
	with the source symmetry filter.

	Parameters
	----------
	state : StateMapFoldingSymmetric
		State supplied by the worker after dequeueing.

	Returns
	-------
	state : StateMapFoldingSymmetric
		The supplied state.
	"""
	return state

def filterAsymmetricFolds(state: StateMapFoldingSymmetric) -> None:
	"""Queue a copied state for background symmetry filtering.

	(AI generated docstring)

	You can use this function to submit a snapshot of `state` without waiting for filtering.
	The function copies `state` with `deepcopy`, enqueues the copy, and returns `None`.

	Parameters
	----------
	state : StateMapFoldingSymmetric
		Current folding state to copy and submit for filtering.

	Examples
	--------
	The module `mapFolding.synthesized.mapFoldingSymmetric.algorithm` uses this call.

		```python
		filterAsymmetricFolds(state)
		```
	"""
	queueFutures.put_nowait(deepcopy(state))

def getSymmetricTotalFolds() -> 形TotalFolds:
	"""Stop initialized workers after queued work and return the accumulated symmetric count.

	(AI generated docstring)

	You can use this function to append one stop object per worker, join every worker, and obtain
	the shared symmetric total after the workers exit.

	Returns
	-------
	symmetricTotalFolds : 形TotalFolds
		Shared accumulated count after all initialized worker threads have exited.

	Examples
	--------
	The module `mapFolding.synthesized.mapFoldingSymmetric.asynchronous` uses this call.

		```python
		getSymmetricTotalFolds()
		```
	"""
	for _thread in boxOfThreads:
		queueFutures.put(STOPsignal)
	for thread in boxOfThreads:
		thread.join()
	return symmetricTotalFolds
