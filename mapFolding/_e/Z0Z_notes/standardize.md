# Standardizing instructions, identifiers, descriptions, and other semiotic elements for clarity

## Replace `k` and `r` in most places with something semantic

It's probably ok or maybe preferable in "iff.py", but other than that, the reader must know the math conventions to understand what `k` and `r` are.

## Replace "trailing operators", "hidden" operators, and semantically void operators

Example

```python
anInteger: int = intInnit([integerNonnegative], 'integerNonnegative', type[int])[0]
```

It is very easy to miss the purpose of the list brackets--or that the indexer is even there. As a comparison, I've replaced
`dict.values()` with `DOTvalues` in many places, and I like it. In `astToolkit`, I have struggled with an indexing problem that
almost certainly has some valuable lessons for this specific index issue. Ironically, or perhaps appropriately, I think the
relevant code is in class `DOT`; no, wait, I think it's in class `Grab`.

## Semiotics: 2^n-dimensional

- Text
  - 2^n-dimensional
  - 2^n-dimensions
- Identifiers, pseudo-identifiers (e.g., function parameters), and file system objects
  - 2上n
  - 2上nDimensional
  - p2上n
  - p2上nDimensional

2^d, 2Dn, p2d[x]: "d" is not obvious to all readers as "dimension". Define d as dimension wherever it appears.

- 2^d-dimensional
- mapShapeIs2上nDimensions
- 2^n-dimensional (semantic similarities with `ndarray`, "n-dimensional array" from NumPy)
- 2^n-dimensions

## Example: OEIS

`def testCaseOeisFormula(request: pytest.FixtureRequest) -> TestCase:`

- "OeisFormula" is a diminutive form of oeisIDbyFormula: NO MOTHERFUCKING DIMINUTIVES.
- "OeisFormula" is referencing a very specific item, the module `oeisIDbyFormula`, and it is not a generalized form that includes
  `oeisIDbyFormula`, which means `oeisIDbyFormula` is used as a proper noun in this case: use the proper noun in the identifier.
- "Oeis" is not a word: use 'oeis' or 'OEIS' but not OeIs, oEIs, oeiS, or Oeis.

## Example of stupid error message

```python
if testCase.oeisID not in dictionaryOEISMapFolding:
    message: str = f"`{testCase.oeisID}` does not define a map shape."
```

The basic thesis of the error message that was triggered by `if testCase.oeisID not in dictionaryOEISMapFolding:` ought to be
"`testCase.oeisID` is not in `dictionaryOEISMapFolding`, therefore ..."

## Identifiers: past to future, LTR; cause to effect, LTR; general to specific, LTR

`mapShapeFromTestCase`

So testCase to mapShape, not mapShape from testCase.
