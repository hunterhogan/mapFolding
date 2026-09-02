from __future__ import annotations

from typing import NamedTuple

class Interval:
	__slots__ = ('attachedTo', 'distal', 'proximal')
	attachedTo: Branch
	proximal: Interval
	distal: Interval

	def __init__(self, attachedTo: Branch, proximal: Interval | None = None, distal: Interval | None = None) -> None:
		self.attachedTo = attachedTo
		self.proximal = proximal or self
		self.distal = distal or self

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

def makeBranch() -> Branch:
	branch: Branch = Branch()
	branch.advance = Branch()
	branch.advance.advance = branch
	return branch

class ToDo(NamedTuple):
	keepGoing: bool
	lineSegment: int
	depth: int
	countAdvance: bool

class Undo(NamedTuple):
	lineSegment: int
	depth: int
	branch: Branch
	interval: Interval
	IllBeBack: Interval
	intervalDistal: Interval
	intervalProximal: Interval

def flowControl(lineSegment: int, depth: int, n: int, *, symmetric: bool) -> ToDo:
	return ToDo(lineSegment < n, lineSegment + 1, depth + 1, (depth != 1) and not (symmetric and (lineSegment == 1)))

def count(lineSegment: int, depth: int, branch: Branch, n: int, *, symmetric: bool) -> int:
	totalPermutations: int = 0
	crossUncrossLIFO: list[tuple[int, int, Branch, Interval] | Undo] = [(lineSegment, depth, branch, branch.intervalComplement.distal)]

	while crossUncrossLIFO:
		frame: tuple[int, int, Branch, Interval] | Undo = crossUncrossLIFO.pop()
		if len(frame) == 4:
			lineSegment, depth, branch, interval = frame  # pyright: ignore[reportAssignmentType]

			if interval is not branch.intervalAdvance:
				crossUncrossLIFO.append(Undo(lineSegment, depth, *crossRoad(branch, interval)))

				是: ToDo = flowControl(lineSegment, depth, n, symmetric=symmetric)
				if 是.keepGoing:
					if 是.countAdvance:
						crossUncrossLIFO.append((是.lineSegment, 是.depth, interval.attachedTo.advance, interval.attachedTo.advance.intervalComplement.distal))
					crossUncrossLIFO.append((是.lineSegment, 是.depth, interval.attachedTo, interval.attachedTo.intervalComplement.distal))
				else:
					totalPermutations += 1
		else:
			crossUncrossLIFO.append(uncrossRoad(frame))
	return totalPermutations

def crossRoad(branch: Branch, interval: Interval) -> tuple[Branch, Interval, Interval, Interval, Interval]:
	IllBeBack: Interval = branch.intervalComplement
	branchAdvance: Branch = makeBranch()
	branch.intervalComplement = branchAdvance.intervalAdvance
	branchAdvance.intervalAdvance = IllBeBack.proximal
	branch.intervalComplement.distal = interval.distal
	branchAdvance.intervalComplement.distal = interval.proximal.distal
	interval.proximal.distal.proximal.distal = branchAdvance.intervalComplement.proximal
	interval.distal.proximal.distal = branch.intervalComplement.proximal
	intervalDistal, interval.attachedTo = _updateBranch(interval.attachedTo, branch)
	intervalProximal, interval.attachedTo.advance = _updateBranch(interval.attachedTo.advance, branchAdvance)
	return (branch, interval, IllBeBack, intervalDistal, intervalProximal)

def _updateBranch(attachedTo: Branch, branch: Branch) -> tuple[Interval, Branch]:
	interval: Interval = Interval(branch, distal=attachedTo.intervalComplement.distal)
	interval.proximal = Interval(branch.advance, proximal=interval, distal=attachedTo.intervalComplement.proximal)
	attachedTo.intervalComplement.distal.proximal.distal = interval.proximal
	attachedTo.intervalComplement.distal = interval
	return interval, attachedTo

def uncrossRoad(是: Undo) -> tuple[int, int, Branch, Interval]:
	for uncross in (是.intervalDistal, 是.intervalProximal):
		uncross.proximal.distal.proximal.distal = uncross.distal
		uncross.distal.proximal.distal = uncross.proximal.distal

	interval: Interval = 是.interval
	interval.proximal.distal.proximal.distal = interval
	interval.distal.proximal.distal = interval.proximal
	interval = interval.distal

	是.branch.intervalComplement = 是.IllBeBack
	return 是.lineSegment, 是.depth, 是.branch, interval

def doTheNeedful(n: int, *, symmetric: bool) -> int:
	lineSegment: int = 0
	depth: int = 0
	branch: Branch = makeBranch()
	lineSegment += 1
	depth += 1
	_updateBranch(branch, makeBranch())
	_updateBranch(branch.advance, makeBranch().advance)

	totalPermutations: int = 1
	是: ToDo = flowControl(lineSegment, depth, n, symmetric=symmetric)
	if 是.keepGoing:
		totalPermutations = count(是.lineSegment, 是.depth, branch, n, symmetric=symmetric)
		if 是.countAdvance:
			totalPermutations += count(是.lineSegment, 是.depth, branch.advance, n, symmetric=symmetric)
	return totalPermutations * (2 - symmetric)
