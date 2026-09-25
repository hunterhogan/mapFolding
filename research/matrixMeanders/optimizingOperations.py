#================== analyze aligned ===================================================================================
#======== if bitsAlfa > 1 and bitsZulu > 1 and (bitsAlfaIsEven or bitsZuluIsEven) =====
"""Find a new test.

Find `bitsAlfa > 1 and bitsZulu > 1 and (bitsAlfaIsEven or bitsZuluIsEven)` without bitsAlfa or
bitsZulu because the stack (or stored data) is huge for this test: it's the most space-expensive step
in the algorithm if numpy.unique is not used.

1. arcCode & 0b11 != 0b11
2. 12 <= arcCode
3. arcCode.bit_length() & 1: if True, 1 < alfa; if False, 1 < zulu
4. Missing: the truth of `1 <` for the other bits group.

is there a super cheap function to know if 0 < bitcount of only odd indexed columns? and even indexed
columns?

- (arcCode & 0b11) == 0b11 IFF (bitsAlfaIsEven or bitsZuluIsEven) is False.
- `bitsAlfa` is even IFF `arcCode` is even.

- `bitsAlfa` > 1, so arcCode's LSB is irrelevant; locatorBits ends with 0b101, so arcCode's 2° LSB is
    irrelevant.
- for `bitsZulu > 1`, `bitsZulu` is `arcCode >> 1`, so arcCode's 2° LSB is irrelevant; locatorBits
    ends with 0b101, so arcCode's 3° LSB is irrelevant.
- If `bitsAlfa > 1 and bitsZulu > 1`, then it follows that `arcCode >= 12`, but not vice versa.
    - 1000: alfa = 0
    - 1001: alfa = 1 !> 1
    - 1010: alfa = 0
    - 1011: alfa = 1 !> 1
    - 1100: alfa = 4, zulu = 4
    - 10100: alfa = 20, zulu = 0

given 12 <= arcCode. arcCode.bit_length() & 1: if True 1 < alfa, if False 1 < zulu.

"""

from __future__ import annotations

arcCode: int = 8

twoBits = arcCode & 0b11


"""bitsAlfaIsEven, bitsZuluIsEven truth table
True	True	Analyze value; == & | bitsAlfa bitsZulu 1 0
True	False	Align bitsAlfa, analyze value
False	True	Align bitsZulu, analyze value
False	False	Skip value; ^ & & bitsAlfa 1 bitsZulu 1
"""
