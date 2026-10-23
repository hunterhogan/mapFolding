"""Explore semi-meander triangle identities and fixed-diagonal formulas.

(AI generated docstring)

You can use this package to evaluate formulas associated with OEIS A400429 [1], derive related
sequence values from supplied triangle rows, and compare formulas with stored counts. The
triangle-cell functions require the needed rows and columns to be present; the functions do
not compute missing cells.

Diagonal Formula Status
-----------------------
The first four fixed-diagonal formulas are proved on their stated domains. The formulas for
diagonals five through eleven agree with known counts but remain conjectural as complete
diagonal formulas. `mapFolding.oeis.A400429._A005315` [2] records domains, initial exceptions,
coefficient representations, and proof references. Agreement printed by `checkFormulas` is a
comparison with available data, not an additional proof.

Contents
--------
Triangle-cell functions
	A000136
		Compute a stamp-folding value from a supplied row sum and its row number.
	A000682
		Compute a semi-meander value from a supplied row sum.
	A005315
		Retrieve a closed-meander value from a supplied first-column cell.
	A005316
		Evaluate the open-meander identity using supplied triangle cells.
	A006661
		Evaluate the A006661 identity using supplied triangle cells.
	A076876
		Evaluate the A076876 identity using supplied triangle cells.
	A077054
		Evaluate the A077054 identity using supplied triangle cells.
	A077460
		Evaluate the A077460 identity using supplied triangle cells and row sums.
A005315 formula family
	A005315of0
		Evaluate the separate base-case expression.
	A005315of1
		Evaluate the first fixed-diagonal expression with a selectable input offset.
	A005315of10
		Evaluate the tenth fixed-diagonal expression with a selectable input offset.
	A005315of11
		Evaluate the eleventh fixed-diagonal expression with a selectable input offset.
	A005315of2
		Evaluate the second fixed-diagonal expression with a selectable input offset.
	A005315of3
		Evaluate the third fixed-diagonal expression with a selectable input offset.
	A005315of4
		Evaluate the fourth fixed-diagonal expression with a selectable input offset.
	A005315of5
		Evaluate the fifth fixed-diagonal expression with a selectable input offset.
	A005315of6
		Evaluate the sixth fixed-diagonal expression with a selectable input offset.
	A005315of7
		Evaluate the seventh fixed-diagonal expression with a selectable input offset.
	A005315of8
		Evaluate the eighth fixed-diagonal expression with a selectable input offset.
	A005315of9
		Evaluate the ninth fixed-diagonal expression with a selectable input offset.
Fixed-diagonal functions
	工diagonal1
		Evaluate the first diagonal at its default offset.
	工diagonal10
		Evaluate the conjectured tenth diagonal at its default offset.
	工diagonal11
		Evaluate the conjectured eleventh diagonal at its default offset.
	工diagonal2
		Evaluate the second diagonal at its default offset.
	工diagonal3
		Evaluate the third diagonal at its default offset.
	工diagonal4
		Evaluate the fourth diagonal at its default offset.
	工diagonal5
		Evaluate the conjectured fifth diagonal at its default offset.
	工diagonal6
		Evaluate the conjectured sixth diagonal at its default offset.
	工diagonal7
		Evaluate the conjectured seventh diagonal at its default offset.
	工diagonal8
		Evaluate the conjectured eighth diagonal at its default offset.
	工diagonal9
		Evaluate the conjectured ninth diagonal at its default offset.
Verification
	checkFormulas
		Print comparisons against known triangle cells, related sequences, and diagonal CSV files.
Variables
	boxOfA005315of
		Select a formula by suffix, with the separate base case at position zero.
	boxOfDiagonals
		Select a fixed-diagonal function, with the first diagonal at position zero.

References
----------
[1] OEIS A400429, semi-meanders by crossings and winding number.
	https://oeis.org/A400429
[2] `mapFolding.oeis.A400429._A005315`
"""
from __future__ import annotations

from mapFolding.oeis.A400429oeisByFormula._A005315 import (
	A005315of0 as A005315of0, A005315of1 as A005315of1, A005315of2 as A005315of2, A005315of3 as A005315of3, A005315of4 as A005315of4,
	A005315of5 as A005315of5, A005315of6 as A005315of6, A005315of7 as A005315of7, A005315of8 as A005315of8, A005315of9 as A005315of9,
	A005315of10 as A005315of10, A005315of11 as A005315of11, boxOfA005315of as boxOfA005315of)
from mapFolding.oeis.A400429oeisByFormula._integrate_fromTriangleCells import (
	A000136 as A000136, A000682 as A000682, A005315 as A005315, A005316 as A005316, A006661 as A006661, A076876 as A076876, A077054 as A077054,
	A077460 as A077460)
from mapFolding.oeis.A400429oeisByFormula._integrate_verify import checkFormulas as checkFormulas
from mapFolding.oeis.A400429oeisByFormula._semi import (
	boxOfDiagonals as boxOfDiagonals, 工diagonal1 as 工diagonal1, 工diagonal2 as 工diagonal2, 工diagonal3 as 工diagonal3, 工diagonal4 as 工diagonal4,
	工diagonal5 as 工diagonal5, 工diagonal6 as 工diagonal6, 工diagonal7 as 工diagonal7, 工diagonal8 as 工diagonal8, 工diagonal9 as 工diagonal9,
	工diagonal10 as 工diagonal10, 工diagonal11 as 工diagonal11)
