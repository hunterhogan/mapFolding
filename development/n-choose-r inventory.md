# n-choose-r terminology: a descriptive inventory; refreshed 2026-10-22

## Status overview

Ordinary coefficient uses and the scalar/basis terminology now have stable names.
Completed entries have been removed. The remaining temporary labels concern Lobb
coefficients; their exact source locations are in
[the annex](n-choose-r%20inventoryAnnex.md).

`theCoefficients` identifies THE coefficients to couple with the Catalan numbers
to discover a closed form of A000682. `工theCoefficients`,
`lookupTheCoefficients`, `theCoefficients.csv`, and `theCoefficients_incomplete.csv`
retain that specific identity.

`catalanMultiplicands` and `binomialMultiplicands` identify the mixed correction
basis multiplicands. The scalar signatures `binomial(top, degree)` and `catalan(n)`,
the basis operations, Dyck names, and combinations enumeration are stable.

## Remaining Lobb-coefficient terminology

- The original coefficients c_(k,i) are THE Catalan-linear coefficients.
- The transformed Lobb coefficients h_(k,r) are a different family. Their temporary
  markers remain in `Z0Z_lookupLobbX`, `Z0Z_extendRowSumXLobb`,
  `Z0Z_pathFilenameCoefficientsLobb`, and `Z0Z_coefficientsLobb.csv`.
- `analyze.Z0Z_extendRowSumXLobb` uses these transformed coefficients as an
  intermediate representation to extend X rows.
- `Z0Z_research/catalanArches/explain/lobb.md` describes the transformation, its
  inverse, and the distinction between original and transformed coefficients.
- The Lobb-number basis itself is A039599:
  L(m,r) = binomial(2*m,m-r) - binomial(2*m,m-r-1).
  Its first argument is the row; its second argument is the offset/column.

## Learning notes retained from the earlier inventory

- ‼️ Lobb: I didn't know this is nCr.
- ‼️ Dyck: I didn't know this is nCr.
- ❓What is Narayana?
- What does "anchor" mean in this context? `kArchetype` is
  max(次k, 次k.sansAntiCatalan), where 次k.sansAntiCatalan = 2*次k-4.
