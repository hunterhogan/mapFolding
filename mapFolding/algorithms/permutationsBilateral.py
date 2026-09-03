from __future__ import annotations

from typing import NamedTuple

class Interval:
	__slots__ = ('distal', 'milepost', 'proximal')
	milepost: Milepost
	proximal: Interval
	distal: Interval

	def __init__(self, milepost: Milepost, proximal: Interval | None = None, distal: Interval | None = None) -> None:
		self.milepost = milepost
		self.proximal = proximal or self
		self.distal = distal or self

class Milepost:
	__slots__ = ('complement', 'interval', 'intervalComplement')
	complement: Milepost
	interval: Interval
	intervalComplement: Interval

	def __init__(self) -> None:
		self.complement = self
		self.interval = Interval(self)
		self.intervalComplement = Interval(self, distal=self.interval)

		self.intervalComplement.proximal = Interval(self, proximal=self.intervalComplement)
		self.interval.proximal = Interval(self, proximal=self.interval, distal=self.intervalComplement.proximal)

# DEVELOPMENT The de facto constructor. Try to put the "loopback" assignments in the real constructor.
def makeMilepost() -> Milepost:
	milepost: Milepost = Milepost()
	milepost.complement = Milepost()
	milepost.complement.complement = milepost
	return milepost

class ToDo(NamedTuple):
	keepCounting: bool
	n下i: int
	countComplement: bool

class Undo(NamedTuple):
	n下i: int
	milepost: Milepost
	interval: Interval
	intervalComplement: Interval
	intervalDistal: Interval
	intervalProximal: Interval

def whatToDo(n下i: int, n: int) -> ToDo:
	return ToDo(n下i < n, n下i := n下i + 1, n下i != 2)

def count(n下i: int, milepost: Milepost, n: int) -> int:
	totalPermutations: int = 0
	toDoUndoLIFO: list[tuple[int, Milepost, Interval] | Undo] = [(n下i, milepost, milepost.intervalComplement.distal)]

	while toDoUndoLIFO:
		work: tuple[int, Milepost, Interval] | Undo = toDoUndoLIFO.pop()
		if len(work) == 3:
			n下i, milepost, interval = work  # pyright: ignore[reportAssignmentType]

			if interval is not milepost.interval:
				toDoUndoLIFO.append(Undo(n下i, *crossRoad(milepost, interval)))

				是: ToDo = whatToDo(n下i, n)
				if 是.keepCounting:
					if 是.countComplement:
						toDoUndoLIFO.append((是.n下i, interval.milepost.complement, interval.milepost.complement.intervalComplement.distal))
					toDoUndoLIFO.append((是.n下i, interval.milepost, interval.milepost.intervalComplement.distal))
				else:
					totalPermutations += 1
		else:
			toDoUndoLIFO.append(uncrossRoad(work))
	return totalPermutations

def crossRoad(milepost: Milepost, interval: Interval) -> tuple[Milepost, Interval, Interval, Interval, Interval]:
	intervalComplement: Interval = milepost.intervalComplement  # NOT a local alias.
	milepostDiagonal: Milepost = makeMilepost()
	milepost.intervalComplement = milepostDiagonal.interval
	milepostDiagonal.interval = intervalComplement.proximal
	milepost.intervalComplement.distal = interval.distal
	milepostDiagonal.intervalComplement.distal = interval.proximal.distal
	interval.proximal.distal.proximal.distal = milepostDiagonal.intervalComplement.proximal
	interval.distal.proximal.distal = milepost.intervalComplement.proximal
	intervalDistal, interval.milepost = _updateMilepost(interval.milepost, milepost)
	intervalProximal, interval.milepost.complement = _updateMilepost(interval.milepost.complement, milepostDiagonal)
	return (milepost, interval, intervalComplement, intervalDistal, intervalProximal)

def _updateMilepost(milepostComplement: Milepost, milepost: Milepost) -> tuple[Interval, Milepost]:
	interval: Interval = Interval(milepost, distal=milepostComplement.intervalComplement.distal)
	interval.proximal = Interval(milepost.complement, proximal=interval, distal=milepostComplement.intervalComplement.proximal)
	milepostComplement.intervalComplement.distal.proximal.distal = interval.proximal
	milepostComplement.intervalComplement.distal = interval
	return interval, milepostComplement

def uncrossRoad(是: Undo) -> tuple[int, Milepost, Interval]:
	for uncross in (是.intervalDistal, 是.intervalProximal):
		uncross.proximal.distal.proximal.distal = uncross.distal
		uncross.distal.proximal.distal = uncross.proximal.distal

	interval: Interval = 是.interval
	interval.proximal.distal.proximal.distal = interval
	interval.distal.proximal.distal = interval.proximal
	interval = interval.distal

	是.milepost.intervalComplement = 是.intervalComplement
	return 是.n下i, 是.milepost, interval

def doTheNeedful(n: int, *, symmetric: bool) -> int:
	totalPermutations: int = 1
	milepost: Milepost = makeMilepost()
	n下i: int = 1

	是: ToDo = whatToDo(n下i, n)
	if 是.keepCounting:
		_updateMilepost(milepost, makeMilepost())
		_updateMilepost(milepost.complement, makeMilepost().complement)

	是: ToDo = whatToDo(n下i, n)
	if 是.keepCounting:
		totalPermutations = count(是.n下i, milepost, n)
		if 是.countComplement:
			totalPermutations += count(是.n下i, milepost.complement, n)
		totalPermutations *= (2 - symmetric)
	return totalPermutations
