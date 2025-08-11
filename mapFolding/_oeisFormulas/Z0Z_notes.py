"""Half-formed thoughts.

I have unintentionally made `bridges -= 1` almost meaningless.

Unlike multidimensional map folding, the computation of curveLocations_sub_i for bridges=p does not need to happen during the
series of computation for bridges=p. Each curveLocations_sub_i produces between no curveLocations, curveLocations_sub_q,
curveLocations_sub_r, curveLocations_sub_s, and curveLocations_sub_t, which are recorded as keys in dictionaryCurveLocations.

`while bridges > 0: bridges -= 1` tacitly attaches metadata to each key in dictionaryCurveLocations: specifically the value of
`bridges`. The computation is not complete until the `bridges` value of each key reaches 0.

Therefore, it is hypothetically possible to use one dictionary and to explicitly track the `bridges` value for each key. In
that scenario, the dictionary is effectively a list of jobs. And instead of being at the mercy of the amount of resources
needed by each decrement, bridges -= 1, we can use well-researched techniques to manage resources and the order of
execution.
"""
