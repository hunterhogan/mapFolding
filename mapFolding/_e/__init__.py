"""Developing elimination-based algorithms."""

# isort: split
from mapFolding._e._semiotics import (
	leafOrigin as leafOrigin, pileOrigin as pileOrigin, 一 as 一, 七 as 七, 三 as 三, 九 as 九, 二 as 二, 五 as 五, 八 as 八, 六 as 六,
	四 as 四, 零 as 零, 首一 as 首一, 首一三 as 首一三, 首一二 as 首一二, 首一二三 as 首一二三, 首三 as 首三, 首二 as 首二, 首二三 as 首二三, 首零 as 首零, 首零一 as 首零一,
	首零一三 as 首零一三, 首零一二 as 首零一二, 首零一二三 as 首零一二三, 首零三 as 首零三, 首零二 as 首零二, 首零二三 as 首零二三)

# isort: split
from mapFolding._e._measure import (
	dimensionFourthNearest首 as dimensionFourthNearest首, dimensionNearestTail as dimensionNearestTail,
	dimensionNearest首 as dimensionNearest首, dimensionSecondNearest首 as dimensionSecondNearest首,
	dimensionThirdNearest首 as dimensionThirdNearest首, howManyDimensionsHaveOddParity as howManyDimensionsHaveOddParity,
	leafInSubHyperplane as leafInSubHyperplane, ptount as ptount, thisIsA2DnMap as thisIsA2DnMap,
	Z0Z_creaseNearestTail as Z0Z_creaseNearestTail, Z0Z_invert as Z0Z_invert,
	Z0Z_sumsOfProductsOfDimensionsNearest首 as Z0Z_sumsOfProductsOfDimensionsNearest首)

# isort: split
from mapFolding._e._dataDynamic import (
	getDictionaryLeafDomains as getDictionaryLeafDomains, getDictionaryPileRanges as getDictionaryPileRanges,
	getDomainDimension一 as getDomainDimension一, getDomainDimension二 as getDomainDimension二,
	getDomainDimension首二 as getDomainDimension首二, getDomain二一零and二一 as getDomain二一零and二一,
	getDomain二零and二 as getDomain二零and二, getDomain首零一二and首一二 as getDomain首零一二and首一二, getDomain首零二and首二 as getDomain首零二and首二,
	getLeafDomain as getLeafDomain, getLeavesCreaseBack as getLeavesCreaseBack, getLeavesCreaseNext as getLeavesCreaseNext,
	getPileRange as getPileRange, getZ0Z_precedence as getZ0Z_precedence, getZ0Z_successor as getZ0Z_successor)

"""Rules for maintaining a valid permutation space:

1. In `leavesPinned`, if `leaf` is not pinned, deconstruct `leavesPinned` by the `pile` domain of `leaf`.
	A. For each `pile` in the domain of `leaf`, if `pile` in `leavesPinned` is not occupied, create a new `PermutationSpace` dictionary by appending `leaf` pinned at `pile` to `leavesPinned`.
	B. Replace `leavesPinned` with the group of newly created `PermutationSpace` dictionaries.
2. In `leavesPinned`, if a `pile` is not pinned, deconstruct `leavesPinned` by the `leaf` range (mathematical range) of `pile`.
	A. For each `leaf` in the range of `pile`, if `leaf` is not already pinned in `leavesPinned`, create a new `PermutationSpace` dictionary by appending `leaf` pinned at `pile` to `leavesPinned`.
	B. Replace `leavesPinned` with the group of newly created `PermutationSpace` dictionaries.
3. Do not overwrite or delete a dictionary's pinned leaves because that could cause the dictionary's permutation space to overlap with a different dictionary's permutation space.
"""

"""Perspective changes and code changes:

- To increment in a dimension means to add 1 in the dimensionIndex. If the current value is 0, then the new value is 1. If the
	current value is 1, then the addition requires "carrying" values to "higher" dimensionIndices.
- The `next` crease, in the sense of `k+1` and `r+1` in the inequalities is simple: given `k` and `dimension`,
	`k1 = bit_flip(k, dimension); k1 = k1 if k1 > k else None`.
	Reminder: this is because I evaluate the four `k < r`, not the eight `k ? r` inequalities *a la* Koehler.
"""

"""The 'meaning' of:
- "CTZ" or `howMany0coordinatesAtTail()` measures trailing zeros.
- `int.bit_length()` essentially measures leading zeros.
- `int.bit_count() - 1` or `howManyDimensionsHaveOddParity()` measures how many ones other than the MSD, but unlike CTZ, it counts
the absolute quantity, not just the consecutive ones relative to the LSD.
"""

"""Habits for better performance:
list:	the elements are changing, and the order matters.
tuple:	the elements are fixed, and the order matters.
set:	the elements are changing, and the order does not matter.
frozenset:	the elements are fixed, and the order does not matter.
iterator: order matters, evaluate as needed, possibly short-circuiting.
"""

"""leaf metadata:
	per dimension:
		for inequality checking:
			next leaf or None
			parity
	domain of leaf
	range of leaves in piles
"""

"""
2d6
(0, 32, 48, 56, 60, 62, 63) = sumsOfProductsOfDimensionsNearest首
(0, 1, 3, 7, 15, 31, 63, 127) = sumsOfProductsOfDimensions
leaf descends from 63 in sumsOfProductsOfDimensionsNearest首
first pile is dimensionsTotal and ascends by addends in sumsOfProductsOfDimensions
leaf63 starts at pile6 = 6+0
leaf62 starts at pile7 = 6+1
leaf60 starts at pile10 = 7+3
leaf56 starts at pile17 = 10+7
leaf48 starts at pile32 = 17+15
leaf32 starts at pile63 = 32+31

2d5
(0, 16, 24, 28, 30, 31)
31, 5+0
30, 5+1
28, 6+3
24, 9+7
16, 16+15
(0, 1, 3, 7, 15, 31, 63)
{0: [0],
 1: [1],
 2: [3, 5, 9, 17],
 3: [2, 7, 11, 13, 19, 21, 25],
 4: [3, 5, 6, 9, 10, 15, 18, 23, 27, 29],
 5: [2, 7, 11, 13, 14, 19, 21, 22, 25, 26, 31],
 6: [3, 5, 6, 9, 10, 15, 17, 18, 23, 27, 29, 30],
 7: [2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 31],
 8: [3, 5, 6, 9, 10, 12, 15, 18, 20, 23, 27, 29, 30],
 9: [2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 10: [3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 27, 29, 30],
 11: [2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 12: [3, 5, 6, 9, 10, 12, 15, 18, 20, 23, 27, 29, 30],
 13: [2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 14: [3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 27, 29, 30],
 15: [2, 4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 16: [3, 5, 6, 9, 10, 12, 15, 18, 20, 23, 24, 27, 29, 30],
 17: [2, 4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 18: [5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30],
 19: [4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 20: [5, 6, 9, 10, 12, 15, 18, 20, 23, 24, 27, 29, 30],
 21: [4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 22: [5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30],
 23: [4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 24: [5, 6, 9, 10, 12, 15, 18, 20, 23, 24, 27, 29, 30],
 25: [4, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 26: [9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30],
 27: [8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 28: [9, 10, 12, 18, 20, 23, 24, 27, 29, 30],
 29: [8, 19, 21, 22, 25, 26, 28],
 30: [17, 18, 20, 24],
 31: [16]}
"""


"""crazy^2
leafPredecessor = state.productsOfDimensions[dimensionNearest首(leaf)] + state.productsOfDimensions[howMany0coordinatesAtTail(leaf)]
print(leafPredecessor == int(bit_flip(0, dimensionNearest首(leaf)).bit_flip(howMany0coordinatesAtTail(leaf))))

sumsOfProductsOfDimensionsNearest首: tuple[int, ...] = Z0Z_sumsOfProductsOfDimensionsNearest首(state)
def Z0Z_inverseIsSmallEnough(leaf: int, pile: int, sumsOfProductsOfDimensionsNearest首: tuple[int, ...] = sumsOfProductsOfDimensionsNearest首) -> bool:
	if leaf in sumsOfProductsOfDimensionsNearest首:
		index首 = sumsOfProductsOfDimensionsNearest首.index(leaf)
		firstPile_bb = state.dimensionsTotal + sum(state.sumsOfProductsOfDimensions[0:state.dimensionsTotal-index首+inclusive])
		anotherFormula = int(bit_flip(0, dimensionNearestTail(leaf) + 1)) + howManyDimensionsHaveOddParity(leaf) - 1 - (leaf == leafOrigin)
		print(firstPile_bb ==anotherFormula, leaf, firstPile_bb, anotherFormula)

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

"""Are these patterns useful?
The list of leaves in the range of pile21 for a 2d6 map:
			[2, 4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31, 35, 37, 38, 41, 42, 44, 47, 49, 50, 52, 55, 56, 59, 61, 62	],
start	step
2		12	[2, 				 14, 				 26, 				 38, 					 50,					 62	],
4		17	[	4,						 21, 							 38,												],
4		24	[	4,										 28,															 62	],
7*		6	[	   7,		 13,	 19,		 25,		 31,	 37,		NOT 43,		 49,		 55,		 61,	],
7		7	[	   7,			 14,	 21,			 28,	 35,			 42,		 49,			 56,			],
7		15	[	   7,						 22,					 37,							 52,					],
8		17	[		  8,						 25,							 42,							 59,		],
8		18	[		  8,							 26,							 44,								 62	],
11		15	[			 11,						 26,					 41,							 56,			],
11		24	[			 11,									 35,											 59,		],
14		21	[					 14,							 35,					  					 56,			],
25		12	[									 25,				 37,					 49,					 61,	],
35		12	[													 35,					 47,					 59,		],
38		6	[															 38,		 44,		 50,		 56,		 62	],
38		9	[															 38,			 47,				 56,			],
41		9	[																 41,				 50,			 59,		],
42		8	[																	 42,				 52,				 62	],
49*		6	[																				 49,		 55,		 61,	],

* If I take leaf7, step 6 to the end, it would generate leaf43, which is wrong. But, leaf49, step 6 to the end is valid.
There are many step patterns that would be valid if the ended at approximately leaf32, such as leaf19, step 3, but I only included leaf7, step 6.

[	   7, 	  11, 13, 14, 19, 21, 22, 25, 26, 28, 31, 		 38, 	 42, 44, 		 50, 52, 	 56,		 62	], # if leaf is odd, then leaf*2, leaf*4, and leaf*8 are in the range.

"""

"""products of dimensions and sums of products emerge from the formulas in `getLeafDomain`.
state = EliminationState((2,) * 6)
domainsOfDimensionOrigins = tuple(getLeafDomain(state, leaf) for leaf in state.productsOfDimensions)[0:-1]
sumsOfDimensionOrigins = tuple(accumulate(state.productsOfDimensions))[0:-1]
sumsOfDimensionOriginsReversed = tuple(accumulate(state.productsOfDimensions[::-1], initial=-state.leavesTotal))[1:None]
for dimensionOrigin, domain, sumOrigins, sumReversed in zip(state.productsOfDimensions, domainsOfDimensionOrigins, sumsOfDimensionOrigins, sumsOfDimensionOriginsReversed, strict=False):
	print(f"{dimensionOrigin:<2}\t{domain.start == sumOrigins = }\t{sumOrigins}\t{sumReversed+2}\t{domain.stop == sumReversed+2 = }")
1       domain.start == sumOrigins = True       1       2       domain.stop == sumReversed+2 = True
2       domain.start == sumOrigins = True       3       34      domain.stop == sumReversed+2 = True
4       domain.start == sumOrigins = True       7      50      domain.stop == sumReversed+2 = True
8       domain.start == sumOrigins = True       15      58      domain.stop == sumReversed+2 = True
16      domain.start == sumOrigins = True       31     62	      domain.stop == sumReversed+2 = True
32      domain.start == sumOrigins = True       63      64      domain.stop == sumReversed+2 = True

(Note to self: in `sumReversed+2`, consider if this is better explained by `sumReversed - descending + inclusive` or something similar.)

The piles of dimension origins (sums of products of dimensions) emerge from the following formulas!

(Note: the function below is included to capture the function as it existed at this point in development. I hope the package has improved/evolved by the time you read this.)
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
leaf1 is a dimension origin: its creases up to [-1], which are leaves 3, 5, 9, 17, come before the dimension origins, 2, 4, 8, 16.

This is due to:
leaf	{dimension: crease}
0			{0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 32} <- dimension origins
1			{1: 3, 2: 5, 3: 9, 4: 17, 5: 33}

If leaf2 were before leaf3, it would interpose the crease from leaf1 to leaf3 in dimension1.

Similarly, leaf2 creases up to [-1], which are leaves 6, 10, 18 come before dimension origins, 4, 8, 16, respectively.
2			{0: 3, 2: 6, 3: 10, 4: 18, 5: 34}

The rule against interposing is so strong it extends to leaf3, which is not a dimension origin, but is the first crease from leaf1.
leaf3 creases up to [-1], which are leaves 7, 11, 19, come before the dimension origins, 4, 8, 16, respectively.
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

Therefore, if `dimensionNearest首(k) <= coordinatesOf0AtTail(r)`, then must `pileOf_k < pileOf_r`
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

I think I can use `leafInSubHyperplane` for the "Start over" equivalences.

for equate the piles:
	for equate the excluderLeaf:
		for equate the excluded leaf:
			make a rule
"""

"""Random observation about progressions within a dimension:
dimension end, 31, dimension origin, 16:
11111 	10000 16
1° crease 10010 18
31 (2, 3) 10011 19 = 18 + 1
31 (1, 3) 10101 21 = 18 + 3
31 (1, 2) 11001 25 = 18 + 7
31			< 33 = 18 + 15: not valid
sums: 0, 1, 3, 7

16 has 4 tail zeros.
16 is a dimension origin: each of the 4 zeros is a "sub-dimension origin" relative to 16.
All odd piles are covered by starting with a sub-dimension origin and adding the sums of products of dimensions.
odds: 17, 19, 21, 23, 25, 27, 29, 31
sums: 0, 1,  3,  7,  15
cf:  16, 17, 19, 23, 31
cf:  18, 19, 21, 25, NA
cf:  20, 21, 23, 27, NA
cf:  22, 23, 25, 29, NA

evens: 16, 18, 20, 22, 24, 26, 28, 30
16 is not in the table
sums: 0, 1,  3,  7,  15
cf:  17, 18, 20, 24, NA
cf:  19, 20, 22, 26, NA
cf:  21, 22, 24, 28, NA
cf:  23, 24, 26, 30, NA

To get the evens, count from the end.
evens: 16, 18, 20, 22, 24, 26, 28, 30
sums 15,  7,  3,  1, 0
cf:  16, 24, 28, 30, 31
cf:  NA, 22, 26, 28, 29
cf:  NA, 20, 24, 26, 27
cf:  NA, 18, 22, 24, 25

"""
