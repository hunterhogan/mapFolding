#================== analyze aligned ===================================================================================
#======== if bitsAlfa > 1 and bitsZulu > 1 and (bitsAlfaIsEven or bitsZuluIsEven) =====
"""NOTE find `bitsAlfa > 1 and bitsZulu > 1 and (bitsAlfaIsEven or bitsZuluIsEven)` without bitsAlfa or bitsZulu.

- `bitsAlfa` is even IFF `arcCode` is even.
- `bitsAlfa` > 1, so arcCode's LSB is irrelevant; locatorBits ends with 0b101, so arcCode's 2° LSB is irrelevant.
- for `bitsZulu > 1`, `bitsZulu` is `arcCode >> 1`, so arcCode's 2° LSB is irrelevant; locatorBits ends with 0b101, so arcCode's 3° LSB is irrelevant.
- If `bitsAlfa > 1 and bitsZulu > 1`, then it follows that `arcCode >= 8`, but not vice versa.
"""
"""NOTE bitsAlfaIsEven, bitsZuluIsEven truth table
True	True	Analyze value; == & | bitsAlfa bitsZulu 1 0
True	False	Align bitsAlfa, analyze value
False	True	Align bitsZulu, analyze value
False	False	Skip value; ^ & & bitsAlfa 1 bitsZulu 1
"""
