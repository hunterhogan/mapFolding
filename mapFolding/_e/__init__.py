"""Developing elimination-based algorithms."""

# isort: split
from mapFolding._e._semiotics import (
	leafOrigin as leafOrigin, pileOrigin as pileOrigin, PinnedLeaves as PinnedLeaves, 一 as 一, 七 as 七, 三 as 三, 九 as 九,
	二 as 二, 五 as 五, 八 as 八, 六 as 六, 四 as 四, 零 as 零, 首一 as 首一, 首一三 as 首一三, 首一二 as 首一二, 首一二三 as 首一二三, 首三 as 首三, 首二 as 首二,
	首二三 as 首二三, 首零 as 首零, 首零一 as 首零一, 首零一三 as 首零一三, 首零一二 as 首零一二, 首零一二三 as 首零一二三, 首零三 as 首零三, 首零二 as 首零二, 首零二三 as 首零二三)

# isort: split
from mapFolding._e._measure import (
	dimensionFourthNearest首 as dimensionFourthNearest首, dimensionNearest首 as dimensionNearest首,
	dimensionSecondNearest首 as dimensionSecondNearest首, dimensionThirdNearest首 as dimensionThirdNearest首,
	howMany0coordinatesAtTail as howMany0coordinatesAtTail,
	howManyDimensionsHaveOddParity as howManyDimensionsHaveOddParity, leafInSubHyperplane as leafInSubHyperplane,
	ptount as ptount)

# isort: split
from mapFolding._e._data import (
	getDictionaryLeafDomains as getDictionaryLeafDomains, getDictionaryPileRanges as getDictionaryPileRanges,
	getDomainDimension一 as getDomainDimension一, getDomainDimension二 as getDomainDimension二,
	getDomainDimension首二 as getDomainDimension首二, getDomain二一零and二一 as getDomain二一零and二一,
	getDomain二零and二 as getDomain二零and二, getDomain首零一二and首一二 as getDomain首零一二and首一二, getDomain首零二and首二 as getDomain首零二and首二,
	getLeafDomain as getLeafDomain, getListLeavesDecrease as getListLeavesDecrease,
	getListLeavesIncrease as getListLeavesIncrease, getPileRange as getPileRange)

"""Perspective changes and code changes:

- To increment in a dimension means to add 1 in the dimensionIndex. If the current value is 0, then the new value is 1. If the
	current value is 1, then the addition requires "carrying" values to "higher" dimensionIndices.
- The `next` crease, in the sense of `k+1` and `r+1` in the inequalities is simple: given `k` and `dimension`,
	`k1 = bit_flip(k, dimension); k1 = k1 if k1 > k else None`.
	NOTE: this is because I evaluate the 4 `k < r`, not the 8 `k ? r` inequalities *a la* Koehler.
"""

"""The 'meaning' of:
- "CTZ" or `howMany0coordinatesAtTail()` measures trailing zeros.
- `int.bit_length()` essentially measures leading zeros.
- `int.bit_count() - 1` or `howManyDimensionsHaveOddParity()` measures how many ones other than the MSD, but unlike CTZ, it counts
the absolute quantity, not just the consecutive ones relative to the LSD.
"""

"""leaf metadata:
	per dimension:
		for inequality checking:
			next leaf or None
			parity
	domain of leaf
	range of leaves in piles
"""

"""Pairs of leaves with low entropy.
Always consecutive:
...3, 2
...16, 48			1/4 leavesTotal

7840 sequences total
Consecutive in most sequences:
6241	...5,4,
6241	...6,7,
6241	...8,40,	1/8 leavesTotal
6241	...56,24,	1/8 leavesTotal
5897	...4,36,	1/16 leavesTotal
5897	...9,8,
5889	...10,11,
5889	...52,20,	1/16 leavesTotal

`getDomain二一零and二一` has the same basic pattern as `getDomain二零and二` with the parity switched.

Interestingly, the 22 pairs of `leaf二一, leaf二一零` in consecutive piles cover 6241 of 7840 foldsTotal for (2,) * 6 maps.
The combined domain is very small, only 76 pairs, but 22 pairs cover 80% and the other 54 pairs only cover 20%. Furthermore,
in the 22 pairs, `leaf二一零` follows `leaf二一`, but in the rest of the domain, `leaf二一` always follows `leaf二一零`.

The same for leaf二零 and leaf二, but reversed.
"""

"""General observations and analyzing pinning options.
6 dimensionsTotal, with equal lengths.
a 2-dimensional plane abstracted in to 4 additional dimensions. Not a cube, hypercube, or "-hedron".
5 products of dimensions.
pile0 is a "corner".
Declare pile0 as the origin.
pile63 is a "corner", and is the "most" opposite corner from pile0.
foldsTotal is divisible by 6! Implementing this includes a side effect that leavesTotal//2 is fixed at pile63 and that leaf0 is fixed at pile0.
foldsTotal is divisible by leavesTotal, so we can pin a leaf to any one pile. We pin leaf0 to the origin by convention.
Implementing both of these results in leaf0 pinned to pile0, leaf1 fixed to pile1, and leaf32 fixed to pile63.
7840 total enumerated sequences * 6! * 2^6 = 361267200 foldsTotal.

Pilings 2 and 62 are important as the first variable pilings: each pile has only 5 possible leaf assignments.

The permutations of these 5 piles produce 5730 of the 7840 sequences.
2	16	32	48	62	:	5730
	16	32	48	62	:	4843
2	16	32	48		:	4843
	16	32	48		:	3425
2	16	32	  	62	:	2947
2	  	32	48	62	:	2947
2	16	  	48	62	:	2897
2	  	32	  	62	:	328
2	  		48	62	:	307
2	  			62	:	14
----------------------------
5	29	30	29	5	:	distinct leaf possibilities

2	17	31	49	62	:	6068
2	17	31	47	62	:	6055
2	15	31	49	62	:	5964
2	17	31	48	62	:	5958
2	17	33	49	62	:	5863
2	15	31	47	62	:	5863
2	15	32	49	62	:	5856
2	3	4	5	6	:	134 sequences

"""

"""products of dimensions and sums of products emerge from `getLeafDomain`
state = EliminationState((2,) * 6)
domainsOfDimensionOrigins = tuple(getLeafDomain(state, leaf) for leaf in state.productsOfDimensions)[0:-1]
sumsOfDimensionOrigins = tuple(accumulate(state.productsOfDimensions))[0:-1]
sumsOfDimensionOriginsReversed = tuple(accumulate(state.productsOfDimensions[::-1], initial=-state.leavesTotal))[1:None]
for dimensionOrigin, domain, sumOrigins, sumReversed in zip(state.productsOfDimensions, domainsOfDimensionOrigins, sumsOfDimensionOrigins, sumsOfDimensionOriginsReversed, strict=False):
	print(f"{dimensionOrigin:<2}\t{domain.start == sumOrigins = }\t{sumOrigins}\t{sumReversed+2}\t{domain.stop == sumReversed+2 = }")
1       domain.start == sumOrigins = True       1       2       domain.stop == sumReversed+2 = True
2       domain.start == sumOrigins = True       3       34      domain.stop == sumReversed+2 = True
4       domain.start == sumOrigins = True       7       50      domain.stop == sumReversed+2 = True
8       domain.start == sumOrigins = True       15      58      domain.stop == sumReversed+2 = True
16      domain.start == sumOrigins = True       31      62      domain.stop == sumReversed+2 = True
32      domain.start == sumOrigins = True       63      64      domain.stop == sumReversed+2 = True

The sums of dimension origins (sums of products of dimensions) emerge from the following formulas!

def getLeafDomain(state: EliminationState, leaf: int) -> range:
	def workhorse(leaf: int, dimensionsTotal: int, mapShape: tuple[int, ...], leavesTotal: int) -> range:
		originPinned =  leaf == leafOrigin
		return range(
					int(bit_flip(0, howMany0coordinatesAtTail(leaf) + 1))									# `start`, first value included in the `range`.
						+ howManyDimensionsHaveOddParity(leaf)
						- 1 - originPinned
					, int(bit_mask(dimensionsTotal) ^ bit_mask(dimensionsTotal - dimensionNearest首(leaf)))	# `stop`, first value excluded from the `range`.
						- howManyDimensionsHaveOddParity(leaf)
						+ 2 - originPinned
					, 2 + (2 * (leaf == 首零(dimensionsTotal)+零))											# `step`
				)
	return workhorse(leaf, state.dimensionsTotal, state.mapShape, state.leavesTotal)
"""

"""if `dimensionNearest首(k) <= coordinatesOf0AtTail(r)`, then must `pileOf_k < pileOf_r`
leaf1 is a dimension origin: its addends up to [-1], which equate to leaves 3, 5, 9, 17, come before the dimension origins, 2, 4, 8, 16.

This is due to:
leaf	{dimension: increase}
0			{0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 32} <- dimension origins
1			{1: 3, 2: 5, 3: 9, 4: 17, 5: 33}

If leaf2 were before leaf3, it would interpose the crease from leaf1 to leaf3 in dimension1.

Similarly, leaf2 addends up to [-1], which equate to leaves 6, 10, 18 come before dimension origins, 4, 8, 16.
2			{0: 3, 2: 6, 3: 10, 4: 18, 5: 34}

The rule against interposing is so strong it extends to leaf3, which is not a dimension origin, but is the first increase from leaf1.
leaf3 addends up to [-1], which equate to leaves 7, 11, 19, come before the dimension origins, 4, 8, 16.
3			{2: 7, 3: 11, 4: 19, 5: 35}

leaf4 is the dimension2 origin and its increases 12 and 20 come before dimension origins 8 and 16.
4			{0: 5, 1: 6, 3: 12, 4: 20, 5: 36}

leaf5, 0b101, 二 + 零, which absolutely has the coordinates of 1 in dimension2, 二, and 1 in dimension0, 零, comes before all multiples of 4.

leaf6, 二 + 一, is the same as leaf5.

leaf7, 二 + 一 + 零, is also the same as leaf5 and leaf6!

leaf9, 三 + 零, comes before the dimension3 origin leaf8, as described above, and before all multiples of 8, or 三.

Furthermore, all leaves between 三+零 and 三+二+一+零, inclusive, come before 三 (8) and its multiples.

The same thing happens at the next dimension, 四. leaves 17-31 all come before 16, 32, and 48. This example is a 6 dimensional
map. Because all leaves less than 32 must come before leaf32, it cannot appear before pile 32. It's fixed at the last pile, of course.

wow.

if `dimensionNearest首(k) <= coordinatesOf0AtTail(r)`, then must `pileOf_k < pileOf_r`
"""

"""Equating pile = leavesTotal // 2 - 1.
dict_keys([2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 31])
dict_keys([2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 31, 35, 37, 38, 41, 42, 47, 49, 50, 55, 59, 61, 62])

dict_keys([2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 31												  ]) # bit_mask(5)
dict_keys([	  2,    		 7, 		11, 	13,     		19, 	21,     	25,     		31]) # right shift 1 or //2
dict_keys([	     											26,	25,	22,	21,		14,	13,		 4,	 2,	  ]) # count from the end: 63 - x.
		1,			11, 13,		19, 21,		25,			35, 37,		41,			49,		55, 59
		31,			26, 25,		22, 21,		19,			14, 13,		11,			 7,		 4,  2
dict_keys([			26, 25,		22, 21, 	19,			14, 13,		11,			 7,		 4,	 2,		  ]) # Double and count from the end.
dict_keys([						11, 13, 14															  ]) # "Start over" at leavesTotal // 4.
dict_keys([												19, 21, 22, 25, 26, 31						  ]) # "Start over" at leavesTotal // 2.
dict_keys([																		25, 26, 31			  ]) # "Start over" at 3 * leavesTotal // 4.

"Counting from the end" is a permutation of the indices applied by bit manipulation.
	pile31 if leavesTotal=64 (see above):
dict_keys([2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 31, 35, 37, 38, 41, 42, 47, 49, 50, 55, 59, 61, 62])
	pile15 if leavesTotal=32, but mapping values based on "counting from the end" (see above):
dict_keys([	     											26,	25,	22,	21,		14,	13,		 4,	 2,	  ]) # count from the end: 63 - x.
	permute by flipping the bits (see below):
														... 26, 25, 22, 21, ... 14, 13,	...	 4,  2
														... 37, 38, 41, 42, ... 49, 50,	...	59, 61

from gmpy2 import bit_mask
for nn in [2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 31][::-1]:
	ii = 6
	mm = bit_mask(ii)
	invert = nn ^ mm
	print(nn, invert, sep='\t-> ')

bin(63) 	= 0b111111
bit_mask(6) = 0b111111

31	-> 32
26	-> 37
25	-> 38
22	-> 41
21	-> 42
19	-> 44
14	-> 49
13	-> 50
11	-> 52
7	-> 56
4	-> 59
2	-> 61

o.m.f.g.

for equate the piles:
	for equate the excluderLeaf:
		for equate the excluded leaf:
			make a rule
"""

