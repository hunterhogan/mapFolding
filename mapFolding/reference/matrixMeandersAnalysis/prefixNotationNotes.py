# ruff: noqa
# ================= analyze aligned ===================================================================================
# ======= if bitsAlpha > 1 and bitsZulu > 1 and (bitsAlphaIsEven or bitsZuluIsEven) =====
"""NOTE find `bitsAlpha > 1 and bitsZulu > 1 and (bitsAlphaIsEven or bitsZuluIsEven)` without bitsAlpha or bitsZulu.
- `bitsAlpha` is even IFF `arcCode` is even.
- `bitsAlpha` > 1, so arcCode's LSB is irrelevant; locatorBits ends with 0b101, so arcCode's 2° LSB is irrelevant.
- for `bitsZulu > 1`, `bitsZulu` is `arcCode >> 1`, so arcCode's 2° LSB is irrelevant; locatorBits ends with 0b101, so arcCode's 3° LSB is irrelevant.
- If `bitsAlpha > 1 and bitsZulu > 1`, then it follows that `arcCode >= 8`, but not vice versa.
"""
"""NOTE bitsAlphaIsEven, bitsZuluIsEven truth table
True	True	Analyze value; == & | bitsAlpha bitsZulu 1 0
True	False	Align bitsAlpha, analyze value
False	True	Align bitsZulu, analyze value
False	False	Skip value; ^ & & bitsAlpha 1 bitsZulu 1
"""
