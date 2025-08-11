from mapFolding._oeisFormulas.A000682 import A000682

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

