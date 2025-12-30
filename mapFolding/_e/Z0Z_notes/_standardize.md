# Standardizing identifiers, descriptions, and other semiotic elements for clarity

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

2^d, 2Dn, p2d[x]: "d" is not obvious to all readers as "dimension". Define d as dimension wherever it appears.

- 2^d-dimensional
- mapShapeIs2上nDimensions
- 2^n-dimensional (semantic similarities with `ndarray`, "n-dimensional array" from NumPy)
- 2^n-dimensions

## Consolidate and organize knowledge in "Elimination.md"

- mapFolding\_e\analysisPython\Z0Z_p2d6.py
- mapFolding\_e\analysisPython\Z0Z_hypothesis.py
- mapFolding\_e\knowledgeDump.py
