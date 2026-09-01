# cython: language_level=3
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, Future

import cython  # pyright: ignore[reportMissingTypeStubs]

@cython.cclass
class Interval:
	__slots__ = ('attachedTo', 'distal', 'proximal')
	attachedTo: Branch
	proximal: Interval
	distal: Interval

	def __init__(self, attachedTo: Branch, proximal: Interval | None = None, distal: Interval | None = None) -> None:
		self.attachedTo = attachedTo
		self.proximal = proximal or self
		self.distal = distal or self

@cython.cclass
class Branch:
	__slots__ = ('advance', 'intervalAdvance', 'intervalComplement')
	advance: Branch
	intervalAdvance: Interval
	intervalComplement: Interval

	def __init__(self) -> None:
		self.advance = self
		self.intervalAdvance = Interval(self)
		self.intervalComplement = Interval(self, distal=self.intervalAdvance)

		self.intervalComplement.proximal = Interval(self, proximal=self.intervalComplement)
		self.intervalAdvance.proximal = Interval(self, proximal=self.intervalAdvance, distal=self.intervalComplement.proximal)

@cython.cfunc
def makeBranch() -> Branch:
	branch: Branch = Branch()
	branch.advance = Branch()
	branch.advance.advance = branch
	return branch

# @cython.cfunc
def countCrossing(lineSegment: cython.Py_ssize_t, branch: Branch, n: cython.Py_ssize_t, depth: cython.Py_ssize_t, *, symmetric: cython.bint) -> int:
	totalPermutations: int = 1
	if lineSegment < n:
		lineSegment += 1
		depth += 1
		countAdvance吗: bool = (depth != 2) and not (symmetric and (lineSegment == 2))
		if depth < min(7, n - 5) and countAdvance吗:
			with ProcessPoolExecutor() as executor:
				claimTicketBranch: Future[int] = executor.submit(countBranch, lineSegment, branch, n, depth, symmetric=symmetric)
				claimTicketAdvance: Future[int] = executor.submit(countBranch, lineSegment, branch.advance, n, depth, symmetric=symmetric)

				totalPermutations = claimTicketBranch.result()
				totalPermutations += claimTicketAdvance.result()
		else:
			totalPermutations = countBranch(lineSegment, branch, n, depth, symmetric=symmetric)
			if countAdvance吗:
				totalPermutations += countBranch(lineSegment, branch.advance, n, depth, symmetric=symmetric)
	depth -= 1
	return totalPermutations

@cython.cfunc
def countBranch(lineSegment: cython.Py_ssize_t, branch: Branch, n: cython.Py_ssize_t, depth: cython.Py_ssize_t, *, symmetric: cython.bint) -> int:
	totalPermutations: int = 0
	interval: Interval = branch.intervalComplement.distal

	while interval is not branch.intervalAdvance:
		IllBeBack: Interval = branch.intervalComplement

		branchAdvance: Branch = makeBranch()
		branch.intervalComplement = branchAdvance.intervalAdvance
		branchAdvance.intervalAdvance = IllBeBack.proximal
		branch.intervalComplement.distal = interval.distal
		branchAdvance.intervalComplement.distal = interval.proximal.distal

		interval.proximal.distal.proximal.distal = branchAdvance.intervalComplement.proximal
		interval.distal.proximal.distal = branch.intervalComplement.proximal

		intervalDistal: Interval = crossLine(interval.attachedTo, branch)
		intervalProximal: Interval = crossLine(interval.attachedTo.advance, branchAdvance)
		totalPermutations += countCrossing(lineSegment, interval.attachedTo, n, depth, symmetric=symmetric)

		uncross: Interval
		for uncross in (intervalDistal, intervalProximal):
			uncross.proximal.distal.proximal.distal = uncross.distal
			uncross.distal.proximal.distal = uncross.proximal.distal

		interval.proximal.distal.proximal.distal = interval
		interval.distal.proximal.distal = interval.proximal
		interval = interval.distal

		branch.intervalComplement = IllBeBack
	return totalPermutations

@cython.cfunc
def crossLine(attachedTo: Branch, branch: Branch) -> Interval:
	interval: Interval = Interval(branch, distal=attachedTo.intervalComplement.distal)
	interval.proximal = Interval(branch.advance, proximal=interval, distal=attachedTo.intervalComplement.proximal)

	attachedTo.intervalComplement.distal.proximal.distal = interval.proximal
	attachedTo.intervalComplement.distal = interval
	return interval

def doTheNeedful(n: int, *, symmetric: bool) -> int:
	lineSegment: cython.Py_ssize_t = 0
	depth: cython.Py_ssize_t = 0
	tree: Branch = makeBranch()
	lineSegment += 1
	depth += 1
	crossLine(tree, makeBranch())
	crossLine(tree.advance, makeBranch().advance)
	return countCrossing(lineSegment, tree, n, depth, symmetric=symmetric) * (2 - symmetric)
