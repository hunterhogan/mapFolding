# Ideas to improve the algorithms

## Bifurcate `PermutationSpace` if a `LeafOptions` has exactly two leaves

This is not a subtle implementation, but it might be useful. After `updateListPermutationSpacePileRangesOfLeaves`, something like
`(any(valfilter(bit_count == 3)), extractUndeterminedPiles, state.listPermutationSpace)` to find all `LeafOptions` with
exactly two leaves, then split the corresponding `PermutationSpace` into two `PermutationSpace` objects, replacing
`LeafOptions` with `int`. Should I then run the new `PermutationSpace` back through
`updateListPermutationSpacePileRangesOfLeaves`? I _feel_ like `notEnoughOpenPiles`, for example, will eliminate some of the new
`PermutationSpace` objects, which is the point.

## Sophisticated bifurcation/separation of `PermutationSpace`

Many relationships cannot be expressed with `LeafOptions`. In a 2^6 map, most of the time, leaf9 and leaf13 can be in any
order, but if leaf13 is in pile3, pile5, or pile7, then leaf9 must precede leaf13. If leaf13 is pinned, `_conditionalPredecessors`
will change the `LeafOptions` and `notEnoughOpenPiles` might disqualify the `PermutationSpace`. Nevertheless, it _might_ be
advantageous to divide the `PermutationSpace` into four dictionaries:

1. pile3: leaf13
2. pile5: leaf13
3. pile7: leaf13
4. At pile3, pile5, or pile7, remove leaf13 from `LeafOptions`.

Then other effects would cascade through the four dictionaries due to other functions.

## Make a 2^n-dimensional version of `thisLeafFoldingIsValid`

The math is far less complex with 2^n-dimensional maps: the computational savings might be multiple orders of magnitude.
