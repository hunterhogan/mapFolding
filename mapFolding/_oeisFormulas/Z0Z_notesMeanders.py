from mapFolding.algorithms.oeisIDbyFormula import A000682

"""

How to A000682:
- Start A000682 for n.
- Find A000560(n-1) in dictionaryCurveLocationsKnown or dictionaryCurveLocationsDiscovered.
- STOP computing.
- Double the total.
That is A000682.


How to A259703:
- Start A000682 for n.
- Find n-1 keys in dictionaryCurveLocationsKnown or dictionaryCurveLocationsDiscovered.
- Descending sort.
That is a A259703 row.

https://oeis.org/A259703

SYMMETRY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

     1;
     1,    1;
     2,    1,    1;
     5,    2,    2,   1;
    12,    5,    4,   2,   1;
    33,   13,   12,   4,   3,   1;
    87,   35,   30,  12,   6,   3,  1;
   252,   98,   90,  32,  21,   6,  4,  1;
   703,  278,  243,  94,  54,  21,  8,  4,  1;
  2105,  812,  745, 270, 175,  57, 32,  8,  5, 1;
  6099, 2385, 2108, 808, 485, 181, 84, 32, 10, 5, 1;


n+1
Row sums are A000682. First column is A000560.
Row sums are A000682. First column is A000560.
Row sums are A000682. First column is A000560.
Row sums are A000682. First column is A000560.
Row sums are A000682. First column is A000560.
Row sums are A000682. First column is A000560.
Row sums are A000682. First column is A000560.
Row sums are A000682. First column is A000560.

     n/a;
     1=    1;
     2=    1+    1;
     5=    2+    2+   1;
    12=    5+    4+   2+   1;
    33=   13+   12+   4+   3+   1;
    87=   35+   30+  12+   6+   3+  1;
   252=   98+   90+  32+  21+   6+  4+  1;
   703=  278+  243+  94+  54+  21+  8+  4+  1;
  2105=  812+  745+ 270+ 175+  57+ 32+  8+  5+ 1;
  6099= 2385+ 2108+ 808+ 485+ 181+ 84+ 32+ 10+ 5+ 1;

print(1==    1)
print(2==    1+    1)
print(5==    2+    2+   1)
print(12==    5+    4+   2+   1)
print(33==   13+   12+   4+   3+   1)
print(87==   35+   30+  12+   6+   3+  1)
print(252==   98+   90+  32+  21+   6+  4+  1)
print(703==  278+  243+  94+  54+  21+  8+  4+  1)
print(2105==  812+  745+ 270+ 175+  57+ 32+  8+  5+ 1)
print(6099== 2385+ 2108+ 808+ 485+ 181+ 84+ 32+ 10+ 5+ 1)


"""

listA259703rowTerms = [1, 1]
n = len(listA259703rowTerms) + 1
rowSum = sum(listA259703rowTerms)
A000682for_n = A000682(n)
print(rowSum == A000682for_n, f"{n = }, {A000682for_n = }, {rowSum = }")

listA259703rowTerms = [2, 1, 1]
n = len(listA259703rowTerms) + 1
rowSum = sum(listA259703rowTerms)
A000682for_n = A000682(n)
print(rowSum == A000682for_n, f"{n = }, {A000682for_n = }, {rowSum = }")

listA259703rowTerms = [5, 2, 2, 1]
n = len(listA259703rowTerms) + 1
rowSum = sum(listA259703rowTerms)
A000682for_n = A000682(n)
print(rowSum == A000682for_n, f"{n = }, {A000682for_n = }, {rowSum = }")

listA259703rowTerms = [12, 5, 4, 2, 1]
n = len(listA259703rowTerms) + 1
rowSum = sum(listA259703rowTerms)
A000682for_n = A000682(n)
print(rowSum == A000682for_n, f"{n = }, {A000682for_n = }, {rowSum = }")

listA259703rowTerms = [33, 13, 12, 4, 3, 1]
n = len(listA259703rowTerms) + 1
rowSum = sum(listA259703rowTerms)
A000682for_n = A000682(n)
print(rowSum == A000682for_n, f"{n = }, {A000682for_n = }, {rowSum = }")

listA259703rowTerms = [87, 35, 30, 12, 6, 3, 1]
n = len(listA259703rowTerms) + 1
rowSum = sum(listA259703rowTerms)
A000682for_n = A000682(n)
print(rowSum == A000682for_n, f"{n = }, {A000682for_n = }, {rowSum = }")

listA259703rowTerms = [252, 98, 90, 32, 21, 6, 4, 1]
n = len(listA259703rowTerms) + 1
rowSum = sum(listA259703rowTerms)
A000682for_n = A000682(n)
print(rowSum == A000682for_n, f"{n = }, {A000682for_n = }, {rowSum = }")

listA259703rowTerms = [703, 278, 243, 94, 54, 21, 8, 4, 1]
n = len(listA259703rowTerms) + 1
rowSum = sum(listA259703rowTerms)
A000682for_n = A000682(n)
print(rowSum == A000682for_n, f"{n = }, {A000682for_n = }, {rowSum = }")

listA259703rowTerms = [2105, 812, 745, 270, 175, 57, 32, 8, 5, 1]
n = len(listA259703rowTerms) + 1
rowSum = sum(listA259703rowTerms)
A000682for_n = A000682(n)
print(rowSum == A000682for_n, f"{n = }, {A000682for_n = }, {rowSum = }")

listA259703rowTerms = [6099, 2385, 2108, 808, 485, 181, 84, 32, 10, 5, 1]
n = len(listA259703rowTerms) + 1
rowSum = sum(listA259703rowTerms)
A000682for_n = A000682(n)
print(rowSum == A000682for_n, f"{n = }, {A000682for_n = }, {rowSum = }")

"""'Hidden' array allocations.

Some numpy operations create temporary arrays in memory that do not have identifiers. I don't know anything about the long-term
affects on memory.

Actually, I don't usually know which operations create temporary arrays. So, I may need to investigate that first.

gc.set_threshold?

"""

"""'1' is the new zero.
https://en.wikipedia.org/wiki/Heaviside_step_function
https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html#numpy.heaviside

`selectGroupAlphaCurves: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.bool_]] = arrayCurveGroups[:, indexGroupAlpha] > numpy.uint64(1)`
AFAIK, there is never a value of zero in arrayCurveGroups[:, indexGroupAlpha] or arrayCurveGroups[:, indexGroupZulu].
Therefore, I _feel_ like there is a more efficient way to get the selector than `> 1`, but I have no idea what that might be.
"""

"""Accuracy.

n=45 crashed trying to allocate (823958692,) during the statement
`uniqueness = numpy.unique_inverse(arrayCurveLocations[:, indexCurveLocations])`

I think that array was the inverse_index array. I estimated 691 million unique curveLocations for n=45 at bridges=18.5 and I
estimate total non-unique curveLocations to be around 120% of unique values on average.

824 million is 119% of 691 million, so my estimates are accurate enough for making design decisions.

WAIT! I wrote the above observation at 4 am or something. I estimated 691 million for n=46, not 45, but the computation was for
n=45. I'm too exhausted right now to analyze all potential implications, but a simple explanation is that 120% is too low. I did
not compute 120%: I looked at some summary data of how often curveLocations were repeated in an array, and I thought, "That's
probably around 120% on average."
"""

"""Half-formed thoughts.

I have unintentionally made `bridges -= 1` almost meaningless.

Unlike multidimensional map folding, the computation of curveLocations_sub_i for bridges=p does not need to happen during the
series of computation for bridges=p. Each curveLocations_sub_i produces between no curveLocations, curveLocations_sub_q,
curveLocations_sub_r, curveLocations_sub_s, and curveLocations_sub_t, which are recorded as keys in dictionaryCurveLocations.

`while bridges > 0: bridges -= 1` tacitly attaches metadata to each key in dictionaryCurveLocations: specifically the value of
`bridges`. The computation is not complete until the `bridges` value of each key reaches 0.

Therefore, it is hypothetically possible to use one dictionary and to explicitly track the `bridges` value for each key. In
that scenario, the dictionary is effectively a list of jobs. And instead of being at the mercy of the amount of resources
needed by each decrement, bridges -= 1, we can use well-researched techniques to manage resources and the order of
execution.
"""

"""NOTE Time comparison for the `selectCurvesXorAtEven` loop: `numpy.uint64` vs `int`
NOTE NOTE NOTE I also improved speed by changing from
`(arrayCurveGroups[:, indexGroupAlpha] << numpy.uint64(2))`
to
`(arrayCurveGroups[:, indexGroupAlpha] << 2)`

`numpy.uint64`:
n =    25      1.40 seconds
n =    26      2.17 seconds
n =    27      3.49 seconds

`int`:
n =    25      0.50 seconds
n =    26      0.79 seconds
n =    27      1.34 seconds

Dramatic, counterintuitive differences in performance.
"""

"""Semi-meanders.

n = 3   `startingCurveLocations` keys = 3
n = 4   `startingCurveLocations` keys = 4
n = 5   `startingCurveLocations` keys = 4
n = 6   `startingCurveLocations` keys = 5
n = 7   `startingCurveLocations` keys = 5
n = 8   `startingCurveLocations` keys = 6
n = 9   `startingCurveLocations` keys = 6
n = 10  `startingCurveLocations` keys = 7
n = 11  `startingCurveLocations` keys = 7
n = 12  `startingCurveLocations` keys = 8
n = 13  `startingCurveLocations` keys = 8
n = 14  `startingCurveLocations` keys = 9
n = 15  `startingCurveLocations` keys = 9
n = 16  `startingCurveLocations` keys = 10
n = 17  `startingCurveLocations` keys = 10
n = 18  `startingCurveLocations` keys = 11
n = 19  `startingCurveLocations` keys = 11
n = 20  `startingCurveLocations` keys = 12
n = 21  `startingCurveLocations` keys = 12
n = 22  `startingCurveLocations` keys = 13
n = 23  `startingCurveLocations` keys = 13
n = 24  `startingCurveLocations` keys = 14
n = 25  `startingCurveLocations` keys = 14
n = 26  `startingCurveLocations` keys = 15
n = 27  `startingCurveLocations` keys = 15
n = 28  `startingCurveLocations` keys = 16
n = 29  `startingCurveLocations` keys = 16
n = 30  `startingCurveLocations` keys = 17
n = 31  `startingCurveLocations` keys = 17
n = 32  `startingCurveLocations` keys = 18
n = 33  `startingCurveLocations` keys = 18
n = 34  `startingCurveLocations` keys = 19
n = 35  `startingCurveLocations` keys = 19
n = 36  `startingCurveLocations` keys = 20
n = 37  `startingCurveLocations` keys = 20
n = 38  `startingCurveLocations` keys = 21
n = 39  `startingCurveLocations` keys = 21
n = 40  `startingCurveLocations` keys = 22
n = 41  `startingCurveLocations` keys = 22
n = 42  `startingCurveLocations` keys = 23
n = 43  `startingCurveLocations` keys = 23
n = 44  `startingCurveLocations` keys = 24
n = 45  `startingCurveLocations` keys = 24
n = 46  `startingCurveLocations` keys = 25
n = 47  `startingCurveLocations` keys = 25
n = 48  `startingCurveLocations` keys = 26
n = 49  `startingCurveLocations` keys = 26
n = 50  `startingCurveLocations` keys = 27
n = 51  `startingCurveLocations` keys = 27
n = 52  `startingCurveLocations` keys = 28
n = 53  `startingCurveLocations` keys = 28
n = 54  `startingCurveLocations` keys = 29
n = 55  `startingCurveLocations` keys = 29
n = 56  `startingCurveLocations` keys = 30
n = 57  `startingCurveLocations` keys = 30
n = 58  `startingCurveLocations` keys = 31
n = 59  `startingCurveLocations` keys = 31
n = 60  `startingCurveLocations` keys = 32
n = 61  `startingCurveLocations` keys = 32
"""

"""
`bridges = 13`
0xaaaaaaaa.bit_length() = 32
0x40000000.bit_length() = 31

`bridges = 29`
0x5000000000000000.bit_length() = 63
0xaaaaaaaaaaaaaaaa.bit_length() = 64
0x5555555555555555.bit_length() = 63

a(41) = 63 bits
print((6664356253639465480).bit_length())
"""
