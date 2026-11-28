"""Count semi-meanders with bilateral interval links.

(AI generated docstring)

You can count ordinary or symmetric semi-meanders without collecting their leaf orders.
Paired mileposts hold the interval structure used to add and undo crossings.

Contents
--------
Functions
	count
		Count completions from a prepared milepost and crossing number.
	crossRoadDiagonally
		Add a crossing and retain the links needed to undo it.
	doTheNeedful
		Initialize bilateral search and return the selected family's count.
	makeMilepost
		Create a milepost with a distinct, mutually linked complement.
	uncrossRoadLaterally
		Undo a crossing and select the next interval to explore.
	whatToDo
		Determine whether a crossing continues and needs its complement branch.

Classes
	Interval
		Store an interval's milepost and neighboring interval links.
	Milepost
		Store complementary milepost and interval references.
	ToDo
		Describe continuation, next crossing number, and complement branching.
	Undo
		Retain the interval references needed to undo a crossing.
"""
from __future__ import annotations

from typing import NamedTuple

class Interval:
	__slots__ = ('distal', 'milepost', 'proximal')
	milepost: Milepost
	proximal: Interval
	distal: Interval

	def __init__(self, milepost: Milepost, proximal: Interval | None = None, distal: Interval | None = None) -> None:
		"""Initialize an interval with a milepost and neighboring links.

		(AI generated docstring)

		Parameters
		----------
		milepost : Milepost
			Milepost associated with the interval.
		proximal : Interval | None = None
			Proximal link. `None` makes the link refer to the new interval itself.
		distal : Interval | None = None
			Distal link. `None` makes the link refer to the new interval itself.
		"""
		self.milepost = milepost
		self.proximal = proximal or self
		self.distal = distal or self

class Milepost:
	__slots__ = ('complement', 'interval', 'intervalComplement')
	complement: Milepost
	interval: Interval
	intervalComplement: Interval

	def __init__(self) -> None:
		"""Initialize a milepost's interval structure with itself as its complement.

		(AI generated docstring)

		You can use `makeMilepost` when the search requires a distinct, mutually linked
		complement rather than this initial self-reference.
		"""
		self.complement = self
		self.interval = Interval(self)
		self.intervalComplement = Interval(self, distal=self.interval)

		self.intervalComplement.proximal = Interval(self, proximal=self.intervalComplement)
		self.interval.proximal = Interval(self, proximal=self.interval, distal=self.intervalComplement.proximal)

# DEVELOPMENT The de facto constructor. Try to put the "loopback" assignments in the real constructor.
def makeMilepost() -> Milepost:
	"""Create a milepost with a distinct, mutually linked complement.

	(AI generated docstring)

	Returns
	-------
	milepost : Milepost
		New milepost whose complement refers back to it, with fresh interval structures.
	"""
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
	interval: Interval
	milepost: Milepost
	intervalComplement: Interval
	intervalDistal: Interval
	intervalProximal: Interval

def whatToDo(n下i: int, n: int) -> ToDo:
	"""Determine continuation and complement branching at a crossing number.

	(AI generated docstring)

	Parameters
	----------
	n下i : int
		Crossing number just reached.
	n : int
		Final crossing number for the search.

	Returns
	-------
	toDo : ToDo
		Continuation is permitted when `n下i < n`. The returned crossing number is
		`n下i + 1`, and complement branching is suppressed when that number is two.
	"""
	return ToDo(n下i < n, n下i := n下i + 1, n下i != 2)

def count(n下i: int, milepost: Milepost, n: int) -> int:
	"""Count all completions from a prepared bilateral search state.

	(AI generated docstring)

	Input Mutation
	--------------
	The linked interval structure changes during search. Each crossing is undone before
	search continues to the next interval.

	Parameters
	----------
	n下i : int
		Crossing number from which to continue.
	milepost : Milepost
		Prepared search structure whose complementary interval list supplies the first branch.
	n : int
		Final crossing number.

	Returns
	-------
	totalPermutations : int
		Number of completed branches, before the entry point's family multiplier.
	"""
	totalPermutations: int = 0
	toDoUndoLIFO: list[tuple[int, Interval, Milepost] | Undo] = [(n下i, milepost.intervalComplement.distal, milepost)]

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

def crossRoadDiagonally(interval: Interval, milepost: Milepost) -> tuple[Interval, Milepost, Interval, Interval, Interval]:
	"""Add a crossing through an interval and retain its undo references.

	(AI generated docstring)

	Input Mutation
	--------------
	The connected interval and milepost links change to represent the new crossing.

	Parameters
	----------
	interval : Interval
		Interval selected for the crossing.
	milepost : Milepost
		Milepost at which the crossing is added.

	Returns
	-------
	undoLinks : tuple[Interval, Milepost, Interval, Interval, Interval]
		Selected interval, supplied milepost, former complementary interval, and two new
		intervals. Together with the crossing number, these form an `Undo` record.
	"""
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
	return (interval, milepost, intervalComplement, intervalDistal, intervalProximal)

def _updateMilepost(milepostComplement: Milepost, milepost: Milepost) -> tuple[Interval, Milepost]:
	"""I use this operation to attach paired intervals to complementary mileposts.

	(AI generated docstring)

	Input Mutation
	--------------
	The complementary interval links of `milepostComplement` include the new interval pair.

	Parameters
	----------
	milepostComplement : Milepost
		Milepost whose complementary interval links receive the new pair.
	milepost : Milepost
		Milepost assigned to the new interval; its complement is assigned to the paired interval.

	Returns
	-------
	interval : Interval
		New interval, retained so its link changes can be undone.
	milepostComplement : Milepost
		The supplied milepost with its complementary interval links updated.
	"""
	interval: Interval = Interval(milepost, distal=milepostComplement.intervalComplement.distal)
	interval.proximal = Interval(milepost.complement, proximal=interval, distal=milepostComplement.intervalComplement.proximal)
	milepostComplement.intervalComplement.distal.proximal.distal = interval.proximal
	milepostComplement.intervalComplement.distal = interval
	return interval, milepostComplement

def uncrossRoadLaterally(是: Undo) -> tuple[int, Interval, Milepost]:
	"""Undo a crossing and select the next interval to explore.

	(AI generated docstring)

	Input Mutation
	--------------
	The links retained in `是` restore the interval structure before the crossing.

	Parameters
	----------
	是 : Undo
		Crossing number and references retained when the crossing was added.

	Returns
	-------
	work : tuple[int, Interval, Milepost]
		Original crossing number, next distal interval, and restored milepost.
	"""
	for uncross in (是.intervalDistal, 是.intervalProximal):
		uncross.proximal.distal.proximal.distal = uncross.distal
		uncross.distal.proximal.distal = uncross.proximal.distal

	interval: Interval = 是.interval
	interval.proximal.distal.proximal.distal = interval
	interval.distal.proximal.distal = interval.proximal

	interval = interval.distal
	是.milepost.intervalComplement = 是.intervalComplement
	return 是.n下i, interval, 是.milepost

def doTheNeedful(n: int, *, symmetric: bool) -> int:
	"""Count ordinary or symmetric semi-meanders with bilateral search.

	(AI generated docstring)

	Parameters
	----------
	n : int
		For ordinary semi-meanders, one less than the number of crossings. For symmetric
		semi-meanders, the number of crossings.
	symmetric : bool
		`True` selects symmetric semi-meanders at index `n`. `False` selects ordinary
		semi-meanders at index `n + 1` and doubles the representative count when `n` exceeds one.

	Returns
	-------
	totalPermutations : int
		Count for the selected family. Indices at most one return one.
	"""
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
