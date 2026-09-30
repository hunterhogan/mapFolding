# Notes about the code

## absurd

`getTotalLeaves`: this check is one-degree short of absurd, but three lines of early absurdity is better than invalid output later. I'd add more checks if I could think of more. Fail early.

## analyzeArcCodesAligned

Implementations that filter or mutate shared input during this analysis must run it last, so later
analyses still receive every original arc code. This implementation preserves `arrayMeanders`,
`bitsAlfa`, and `bitsZulu`, performing its transformations in `bitsAlfaStack` and `toPrepArea`.
Running the aligned analysis first is therefore intentional.

## arrayWorkbench

`arrayWorkbench` is one disk-backed allocation partitioned into named views for `bitsAlfa`,
`bitsZulu`, and reusable scratch space, `toPrepArea`. Most NumPy operations target these views with
`out=`, reusing the workbench instead of allocating a new full-length result. Slice assignment also
writes through a view; assigning a new object to a view name would merely rebind the name.

`ShapeArray` and `ShapeSlicer` centralize the physical layout, so axes or scratch lanes can be
rearranged without changing the analysis's semantic access names.

## pinning

The ONLY valid way to pin a `Leaf` in a `PermutationSpace` or `Folding` is to call a method of `PermutationSpace`.

## sorted

`PermutationSpace.addMissingChoicesLeaf()`: `sorted` overrides the insertion order and sorts based on `Pile` index. This is partially "defensive" in the sense that it is a consistent, logical, expected order, and may prevent odd results if another subroutine didn't guarantee the order when it ought to have. I'm hoping it improves efficiency, too.

## sortingDimensions

I previously sorted the dimensions for a few reasons that may or may not be valid:

1. After empirical testing, I believe that (2,10), for example, computes significantly faster than (10,2).
2. Standardization, generally.
3. If I recall correctly, after empirical testing, I concluded that sorted dimensions always leads to
non-negative values in the connection graph, but if the dimensions are not in ascending order of magnitude,
the connection graph might have negative values, which as far as I know, is not an inherent problem, but the
negative values propagate into other data structures, which requires the datatypes to hold negative values,
which means I cannot optimize the bit-widths of the datatypes as easily. (And optimized bit-widths helps with
performance.)

Furthermore, now that the package includes OEIS A000136, 1 x N stamps/maps, sorting could distort results.

## TypeAlias

Use `type` by default. Switch to `TypeAlias` whenever it promotes self-documenting code, especially through semiotics. Examples, I prefer `isinstance(x, ChoicesLeaf)` to `isinstance(x, mpz)`; `dimension = DimensionIndex(2)` is more self-documenting than `dimension = int(2)`.

## unambiguousInput

Be nice to callers at the input boundary. When a value has one clear interpretation but arrives in
the wrong Python type, normalize the representation before validating the value. An integer written
as a `str`, such as `'7'`, or supplied as an integral `float`, such as `7.0`, should not halt a
computation solely because of the representation. Do not guess an intended value or round a
fractional value to make the input acceptable.

In [`oeisIDfor_n`](../mapFolding/oeis/_byID.py), values that are not already an `int` pass through
`hunterMakesPy.parseParameters.intInnit`. The function adopts the converted value when the result
contains exactly one integer, then requires a non-negative index and checks the sequence's offset.
Normalization preserves these domain checks; it does not replace them. Unsupported input still
raises an error.

Use the existing conversion and normalization functions in `hunterMakesPy.parseParameters` and
`hunterMakesPy.dataStructures` where applicable instead of implementing another parser.
