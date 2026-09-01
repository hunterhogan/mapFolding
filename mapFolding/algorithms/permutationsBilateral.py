"""Count semi-meanders and symmetric semi-meanders with a mirrored branch structure.

(AI generated docstring)

You can use this module to count semi-meanders and symmetric semi-meanders using a modified version of
Sawada's and Li's [1] permutation algorithm. This module re-expresses the paper's mutable node tree
with mirrored `Branch` and `Interval` abstractions so the same splice and backtracking operations work
from either end of the permutation.

Contents
--------
Classes
    Branch
        Represent one side of a mirrored branch pair.
    Interval
        Represent one mutable interval in the semi-meander branch structure.

Functions
    countBranch
        Count completions reachable from the intervals attached to `branch`.
    countCrossing
        Count completions after crossing the line (the road) at `lineSegment`.
    crossLine
        Attach each new `Interval` created by crossing the line.
    doTheNeedful
        Count semi-meanders or symmetric semi-meanders of order `n`.
    makeBranch
        Create a mirrored pair of `Branch` records.

References
----------
[1] Sawada, J., and Li, R. (2012). Stamp Foldings, Semi-meanders, and Open Meanders:
    Fast Generation Algorithms. The Electronic Journal of Combinatorics, 19(2), P43.
    https://doi.org/10.37236/2404
"""
from __future__ import annotations

from typing import Self

# SEMIOTICS I'm not satisfied with "interval." On the one hand, it is literally an interval, not
# merely analogous to an interval. That is a STRONG reason to use the term: the word means what it is
# and is what it means.

# On the other hand, it's an interval of the line (boring), not the curve (what we are studying), and
# the ends of the interval are ostensibly defined by two intersections of the curve with the line--but
# you can't know anything about the two intersections if you don't know anything about at least one
# other interval. It's not really a space between two points: it's a space between This Point--which
# is or is not a point ending an adjacent interval, which in turn has a second point that is or is not
# a point ending a different adjacent interval, until the series ends--and That Point, which is or is
# not a point ending an adjacent interval, which in turn has a second point that is or is not a point
# ending a different adjacent interval, until the series ends. It's "interval-ness" doesn't tell us
# anything.

# I despise "gap".
class Interval:
	"""Represent one mutable interval in the semi-meander branch structure.

	(AI generated docstring)

	You can use this class to store one valid interval between adjacent permutation entries in the
	mutable linked structure. Each `Interval` stores the `Branch` that owns the interval and the
	neighboring `Interval` records reached through the proximal and distal links. Self-referential
	links mark sentinel positions in empty interval chains.

	Attributes
	----------
	attachedTo : Branch
		The `Branch` that currently owns this `Interval`.
	proximal : Interval
		The neighboring `Interval` reached by following the proximal link.
	distal : Interval
		The neighboring `Interval` reached by following the distal link.
	"""
	# SEMIOTICS "distal" and "proximal" are excellent, but "attached" is focused on the data
	# structure, not the problem. The semiotic _system_ needs to be about the programming or the
	# problem, not a pastiche.
	__slots__ = ('attachedTo', 'distal', 'proximal')
	attachedTo: Branch
	proximal: Interval
	distal: Interval

	def __init__(self, attachedTo: Branch, proximal: Interval | None = None, distal: Interval | None = None) -> None:
		"""I use this to initialize one interval record and its optional neighbors.

		(AI generated docstring)

		You can use this initializer to create a standalone sentinel `Interval` or to splice a new
		`Interval` between existing `proximal` and `distal` neighbors. When `proximal` or `distal`
		is `None`, this initializer replaces that missing neighbor with `self`.

		Parameters
		----------
		attachedTo : Branch
			The `Branch` that will own the new `Interval`.
		proximal : Interval | None = None
			The `Interval` to store in `self.proximal`. When `proximal` is `None`,
			`self.proximal` becomes `self`.
		distal : Interval | None = None
			The `Interval` to store in `self.distal`. When `distal` is `None`,
			`self.distal` becomes `self`.
		"""
		self.attachedTo = attachedTo
		self.proximal = proximal or self
		self.distal = distal or self

class Branch:
	"""Represent one side of a mirrored branch pair.

	(AI generated docstring)

	You can use this class to model the mutable node abstraction used by the semi-meander counter.
	This class is indifferent to leftness and rightness. The algorithm can advance from the head,
	tail, left, or right of the permutation by switching between one `Branch` and `branch.advance`.
	In the same mirrored way, `intervalAdvance` complements `intervalComplement`.

	Attributes
	----------
	advance : Branch
		The mirrored `Branch` that lets the algorithm traverse the same structure from the opposite
		side.
	intervalAdvance : Interval
		The sentinel `Interval` on the side reached when the algorithm advances through this
		`Branch`.
	intervalComplement : Interval
		The mirrored sentinel `Interval` paired with `intervalAdvance`.
	"""
	__slots__ = ('advance', 'intervalAdvance', 'intervalComplement')
	advance: Branch
	intervalAdvance: Interval
	intervalComplement: Interval

	def __init__(self) -> None:
		self.advance = self
		# SEMIOTICS "advance": I almost always read it as the wrong part of speech. "complement": it's
		# almost impossible to read it as meaning "the complement of advance".
		self.intervalAdvance = Interval(self)
		self.intervalComplement = Interval(self, distal=self.intervalAdvance)

		self.intervalComplement.proximal = Interval(self, proximal=self.intervalComplement)
		self.intervalAdvance.proximal = Interval(self, proximal=self.intervalAdvance, distal=self.intervalComplement.proximal)

def makeBranch() -> Branch:
	"""Create a mirrored pair of `Branch` records.

	(AI generated docstring)

	You can use this function to allocate the two `Branch` records that form one mirrored pair. The
	returned `Branch` points to its mirror through `advance`, and the mirror points back to the
	returned `Branch`.

	Returns
	-------
	branch : Branch
		One `Branch` in a newly created mirrored pair.
	"""
	branch: Branch = Branch()
	branch.advance = Branch()
	branch.advance.advance = branch
	return branch

def countCrossing(lineSegment: int, branch: Branch, n: int, depth: int, *, symmetric: bool) -> int:
	"""Count completions after adding one crossing at `branch`.

	(AI generated docstring)

	You can use this function to advance the semi-meander order by one crossing and then count every
	completion reachable from `branch`. When `symmetric` is `True`, this function skips the mirrored
	second crossing so the result counts symmetric semi-meanders instead of all semi-meanders.

	Parameters
	----------
	lineSegment : int
		The current crossing label and current semi-meander order before this step.
	branch : Branch
		The active `Branch` from which the next crossing is explored.
	n : int
		The target semi-meander order.
	depth : int
		The current depth of the active `Branch` in the mirrored branch structure.
	symmetric : bool
		Whether to count only the reflective class obtained by forcing the second crossing to one side.

	Returns
	-------
	totalPermutations : int
		The number of valid completions reachable from the updated state.
	"""
	totalPermutations: int = 1
	if lineSegment < n:
		lineSegment += 1
		depth += 1
		totalPermutations = countBranch(lineSegment, branch, n, depth, symmetric=symmetric)
		if (depth != 2) and not (symmetric and (lineSegment == 2)):
			totalPermutations += countBranch(lineSegment, branch.advance, n, depth, symmetric=symmetric)
	depth -= 1
	return totalPermutations

def countBranch(lineSegment: int, branch: Branch, n: int, depth: int, *, symmetric: bool) -> int:
	"""Count completions reachable from the intervals attached to `branch`.

	(AI generated docstring)

	You can use this function to iterate through each currently valid interval on `branch`, splice in
	one new mirrored branch pair, recurse, and then restore the linked structure. This function
	mutates the existing `Branch` and `Interval` records in place so backtracking does not require a
	copy of the whole semi-meander state.

	Parameters
	----------
	lineSegment : int
		The current crossing label and current semi-meander order.
	branch : Branch
		The active `Branch` whose valid intervals are being crossed.
	n : int
		The target semi-meander order.
	depth : int
		The current depth of the active `Branch` in the mirrored branch structure.
	symmetric : bool
		Whether the recursive count should suppress the mirrored second crossing.

	Returns
	-------
	totalPermutations : int
		The number of valid completions reachable from every interval on `branch`.
	"""
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

		for uncross in (intervalDistal, intervalProximal):
			uncross.proximal.distal.proximal.distal = uncross.distal
			uncross.distal.proximal.distal = uncross.proximal.distal

		interval.proximal.distal.proximal.distal = interval
		interval.distal.proximal.distal = interval.proximal
		interval = interval.distal

		branch.intervalComplement = IllBeBack
	return totalPermutations

def crossLine(attachedTo: Branch, branch: Branch) -> Interval:
	"""Insert a new crossing into the interval chain at `attachedTo`.

	(AI generated docstring)

	You can use this function to splice one new crossing into the mutable interval chain owned by
	`attachedTo`. The new crossing uses `branch` and `branch.advance` as the mirrored branch pair,
	and this function returns the inserted `Interval` that callers later use to undo the splice in
	constant time.

	Parameters
	----------
	attachedTo : Branch
		The existing `Branch` whose complement chain receives the new crossing.
	branch : Branch
		The new mirrored `Branch` pair whose intervals become the inserted crossing.

	Returns
	-------
	insertedInterval : Interval
		The first inserted `Interval` record for the new crossing.
	"""
	interval: Interval = Interval(branch, distal=attachedTo.intervalComplement.distal)
	interval.proximal = Interval(branch.advance, proximal=interval, distal=attachedTo.intervalComplement.proximal)

	attachedTo.intervalComplement.distal.proximal.distal = interval.proximal
	attachedTo.intervalComplement.distal = interval
	return interval

def doTheNeedful(n: int, *, symmetric: bool) -> int:
	"""Count semi-meanders or symmetric semi-meanders of order `n`.

	(AI generated docstring)

	You can use this function to count the Sawada and Li semi-meander families [1] with the mirrored
	`Branch` and `Interval` abstractions defined in this module. This function seeds the compulsory
	initial crossings, performs the recursive count, and doubles the asymmetric count when
	`symmetric` is `False`.

	Parameters
	----------
	n : int
		The target semi-meander order.
	symmetric : bool
		Whether to count only symmetric semi-meanders instead of all semi-meanders.

	Returns
	-------
	totalPermutations : int
		The number of semi-meanders of order `n`. When `symmetric` is `True`, `totalPermutations` counts the
		symmetric semi-meanders of order `n`.

	See Also
	--------
	`mapFolding.algorithms.permutations.doTheNeedful`
		Count the broader stamp-folding and meander family through one dispatcher.

	References
	----------
	[1] Sawada, J., and Li, R. (2012). Stamp Foldings, Semi-meanders, and Open Meanders:
		Fast Generation Algorithms. The Electronic Journal of Combinatorics, 19(2), P43.
		https://doi.org/10.37236/2404
	"""
	lineSegment: int = 0
	depth: int = 0
	tree: Branch = makeBranch()
	lineSegment += 1
	# SEMIOTICS "depth" of the recursion focuses on the program instead of the problem. I think this
	# corresponds to the "index" of a value in the permutation. Is there a technical term for that?
	depth += 1
	crossLine(tree, makeBranch())
	crossLine(tree.advance, makeBranch().advance)
	return countCrossing(lineSegment, tree, n, depth, symmetric=symmetric) * (2 - symmetric)
