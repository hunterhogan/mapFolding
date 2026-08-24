# DOCUMENT
#ruff: file-ignore[undocumented-public-class]
from __future__ import annotations

from typing import Self

class Interval:
	# DOCUMENT
	__slots__ = ('attachedTo', 'distal', 'proximal')

	def __init__(self, attachedTo: Branch, proximal: Interval | None = None, distal: Interval | None = None) -> None:
		self.attachedTo: Branch = attachedTo
		self.proximal: Interval | Self = proximal or self
		self.distal: Interval | Self = distal or self

class Branch:
	# DOCUMENT
	__slots__ = ('advance', 'intervalAdvance', 'intervalComplement')

	def __init__(self) -> None:
		self.advance: Branch = self
		self.intervalAdvance: Interval = Interval(self)
		self.intervalComplement: Interval = Interval(self, distal=self.intervalAdvance)

		self.intervalComplement.proximal = Interval(self, proximal=self.intervalComplement)
		self.intervalAdvance.proximal = Interval(self, proximal=self.intervalAdvance, distal=self.intervalComplement.proximal)

def makeBranch() -> Branch:
	# DOCUMENT
	branch: Branch = Branch()
	branch.advance = Branch()
	branch.advance.advance = branch
	return branch

def countCrossing(lineSegment: int, branch: Branch, n: int, depth: int, *, symmetric: bool) -> int:
	# DOCUMENT
	total: int = 1
	if lineSegment < n:
		lineSegment += 1
		depth += 1
		total = countBranch(lineSegment, branch, n, depth, symmetric=symmetric)
		if (depth != 2) and not (symmetric and (lineSegment == 2)):
			total += countBranch(lineSegment, branch.advance, n, depth, symmetric=symmetric)
	depth -= 1
	return total

def countBranch(lineSegment: int, branch: Branch, n: int, depth: int, *, symmetric: bool) -> int:
	# DOCUMENT
	total: int = 0
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
		total += countCrossing(lineSegment, interval.attachedTo, n, depth, symmetric=symmetric)

		for uncross in (intervalDistal, intervalProximal):
			uncross.proximal.distal.proximal.distal = uncross.distal
			uncross.distal.proximal.distal = uncross.proximal.distal

		interval.proximal.distal.proximal.distal = interval
		interval.distal.proximal.distal = interval.proximal
		interval = interval.distal

		branch.intervalComplement = IllBeBack
	return total

def crossLine(attachedTo: Branch, branch: Branch) -> Interval:
	# DOCUMENT
	interval: Interval = Interval(branch, distal=attachedTo.intervalComplement.distal)
	interval.proximal = Interval(branch.advance, proximal=interval, distal=attachedTo.intervalComplement.proximal)

	attachedTo.intervalComplement.distal.proximal.distal = interval.proximal
	attachedTo.intervalComplement.distal = interval
	return interval

def doTheNeedful(n: int, *, symmetric: bool) -> int:
	# DOCUMENT
	tree: Branch = makeBranch()
	lineSegment: int = 0
	depth: int = 0
	crossLine(tree, makeBranch())
	crossLine(tree.advance, makeBranch().advance)
	lineSegment += 1
	depth += 1
	return countCrossing(lineSegment, tree, n, depth, symmetric=symmetric) * (2 - symmetric)
