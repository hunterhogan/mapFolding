from __future__ import annotations

from typing import NamedTuple

class Interval:
	__slots__ = ('attachedTo', 'distal', 'proximal')
	# SEMIOTICS If "Interval" is an element in a doubly linked list, then this attribute is the data.
	# Side. RoadSide.
	attachedTo: Branch
	proximal: Interval
	distal: Interval

	def __init__(self, attachedTo: Branch, proximal: Interval | None = None, distal: Interval | None = None) -> None:
		self.attachedTo = attachedTo
		self.proximal = proximal or self
		self.distal = distal or self

# SEMIOTICS If the road is west-east, then maybe a "Branch" is either north or south. PlanarHalf.
# Boundary. Math terms: dart, half-edge.
class Branch:
	__slots__ = ('complement', 'interval', 'intervalComplement')
	complement: Branch
	interval: Interval
	intervalComplement: Interval

	def __init__(self) -> None:
		self.complement = self
		self.interval = Interval(self)
		self.intervalComplement = Interval(self, distal=self.interval)

		self.intervalComplement.proximal = Interval(self, proximal=self.intervalComplement)
		self.interval.proximal = Interval(self, proximal=self.interval, distal=self.intervalComplement.proximal)

def makeBranch() -> Branch:
	branch: Branch = Branch()
	branch.complement = Branch()
	branch.complement.complement = branch
	return branch

class ToDo(NamedTuple):
	keepCounting: bool
	n下i: int
	countComplement: bool

class Undo(NamedTuple):
	n下i: int
	branch: Branch
	interval: Interval
	intervalComplement: Interval
	intervalDistal: Interval
	intervalProximal: Interval

def whatToDo(n下i: int, n: int) -> ToDo:
	return ToDo(n下i < n, n下i := n下i + 1, n下i != 2)

def count(n下i: int, branch: Branch, n: int) -> int:
	totalPermutations: int = 0
	toDoUndoLIFO: list[tuple[int, Branch, Interval] | Undo] = [(n下i, branch, branch.intervalComplement.distal)]

	while toDoUndoLIFO:
		work: tuple[int, Branch, Interval] | Undo = toDoUndoLIFO.pop()
		if len(work) == 3:
			n下i, branch, interval = work  # pyright: ignore[reportAssignmentType]

			if interval is not branch.interval:
				toDoUndoLIFO.append(Undo(n下i, *crossRoad(branch, interval)))

				是: ToDo = whatToDo(n下i, n)
				if 是.keepCounting:
					if 是.countComplement:
						toDoUndoLIFO.append((是.n下i, interval.attachedTo.complement, interval.attachedTo.complement.intervalComplement.distal))
					toDoUndoLIFO.append((是.n下i, interval.attachedTo, interval.attachedTo.intervalComplement.distal))
				else:
					totalPermutations += 1
		else:
			toDoUndoLIFO.append(uncrossRoad(work))
	return totalPermutations

def crossRoad(branch: Branch, interval: Interval) -> tuple[Branch, Interval, Interval, Interval, Interval]:
	intervalComplement: Interval = branch.intervalComplement
	branchAdvance: Branch = makeBranch()
	branch.intervalComplement = branchAdvance.interval
	branchAdvance.interval = intervalComplement.proximal
	branch.intervalComplement.distal = interval.distal
	branchAdvance.intervalComplement.distal = interval.proximal.distal
	interval.proximal.distal.proximal.distal = branchAdvance.intervalComplement.proximal
	interval.distal.proximal.distal = branch.intervalComplement.proximal
	intervalDistal, interval.attachedTo = _updateBranch(interval.attachedTo, branch)
	intervalProximal, interval.attachedTo.complement = _updateBranch(interval.attachedTo.complement, branchAdvance)
	return (branch, interval, intervalComplement, intervalDistal, intervalProximal)

def _updateBranch(attachedTo: Branch, branch: Branch) -> tuple[Interval, Branch]:
	interval: Interval = Interval(branch, distal=attachedTo.intervalComplement.distal)
	interval.proximal = Interval(branch.complement, proximal=interval, distal=attachedTo.intervalComplement.proximal)
	attachedTo.intervalComplement.distal.proximal.distal = interval.proximal
	attachedTo.intervalComplement.distal = interval
	return interval, attachedTo

def uncrossRoad(是: Undo) -> tuple[int, Branch, Interval]:
	for uncross in (是.intervalDistal, 是.intervalProximal):
		uncross.proximal.distal.proximal.distal = uncross.distal
		uncross.distal.proximal.distal = uncross.proximal.distal

	interval: Interval = 是.interval
	interval.proximal.distal.proximal.distal = interval
	interval.distal.proximal.distal = interval.proximal
	interval = interval.distal

	是.branch.intervalComplement = 是.intervalComplement
	return 是.n下i, 是.branch, interval

def doTheNeedful(n: int, *, symmetric: bool) -> int:
	totalPermutations: int = 1
	branch: Branch = makeBranch()
	n下i: int = 1

	是: ToDo = whatToDo(n下i, n)
	if 是.keepCounting:
		_updateBranch(branch, makeBranch())
		_updateBranch(branch.complement, makeBranch().complement)

	是: ToDo = whatToDo(n下i, n)
	if 是.keepCounting:
		totalPermutations = count(是.n下i, branch, n)
		if 是.countComplement:
			totalPermutations += count(是.n下i, branch.complement, n)
		totalPermutations *= (2 - symmetric)
	return totalPermutations
