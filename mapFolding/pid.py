"""For troubleshooting, special identifiers:

- Where possible, identifiers match the working non-JAX version: https://github.com/hunterhogan/mapFolding/blob/68a1d7f900caa92916edaba6c2a6dcfd229b3c3a/mapFolding/lovelace.py
- `dynamicHubris_init_val`: parameter `init_val` in `jax.lax.while_loop`, "of type ``a``"
- `dynamicHubris_output_val`: return value of `jax.lax.while_loop`, "of type ``a``"
- `a`: within the while loop, the `dynamicHubris` "of type ``a``", which must be explicitly handled in JAX-style functions
- Sometimes, some conditional statements must be given an identifier, such as "cond_fun" in `jax.lax.while_loop`
    and the predicate in `jax.lax.cond`. The identifier attempts to represent the logic of the conditional
    statement. So `l > 0`, for example, could be `l_greaterThan_0`.

NOTE jax/_src/lax/control_flow/loops.py
```
  def while_loop(
    cond_fun: (T@while_loop) -> BooleanNumeric,
    body_fun: (T@while_loop) -> T@while_loop,
    init_val: T@while_loop
  ) -> T@while_loop
```

 .. code-block:: haskell
  Args:
    cond_fun: function of type ``a -> Bool``.
    body_fun: function of type ``a -> a``.
    init_val: value of type ``a``, a type that can be a scalar, array, or any
      pytree (nested Python tuple/list/dict) thereof, representing the initial
      loop a value.

  Returns:
    The output from the final iteration of body_fun, of type ``a``.

NOTE jax.lax.cond, effectively:
```
  def cond(
    pred,
    true_fun: Callable,
    false_fun: Callable,
    *operands
  ):
```

Which is equivalent to:

```python
  def cond(pred, true_fun, false_fun, *operands):
    if pred:
      return true_fun(*operands)
    else:
      return false_fun(*operands)
```

See jax/_src/lax/control_flow/conditionals.py
"""
from typing import TypedDict
import jax
"""ideas:
- revisit optimizing dtype; except foldingsSubtotal, int8 is (probably) large enough
- revisit mapFoldingPathDivisions; n.b., only tried with `+= 1` not `+= leavesTotal`
"""

def spoon(taskDivisions: jax.Array, arrayIndicesTask: jax.Array, leavesTotal: jax.Array, dimensionsTotal: jax.Array, D: jax.Array):

    class DynamicHubris(TypedDict):
        A: jax.Array
        B: jax.Array
        count: jax.Array
        foldingsSubtotal: jax.Array
        g: jax.Array
        gap: jax.Array
        gapter: jax.Array
        l: jax.Array

    def hubris(taskIndex: jax.Array):
        dynamicHubris_init_val =  DynamicHubris(
            A                = jax.numpy.zeros(leavesTotal + 1,       dtype=jax.numpy.int64),
            B                = jax.numpy.zeros(leavesTotal + 1,       dtype=jax.numpy.int64),
            count            = jax.numpy.zeros(leavesTotal + 1,       dtype=jax.numpy.int64),
            foldingsSubtotal = jax.numpy.zeros(              1,       dtype=jax.numpy.int64),
            g                = jax.numpy.zeros(              1,       dtype=jax.numpy.int64),
            gap              = jax.numpy.zeros((leavesTotal **2) + 1, dtype=jax.numpy.int64),
            gapter           = jax.numpy.zeros(leavesTotal + 1,       dtype=jax.numpy.int64),
            l                = jax.numpy.ones(               1,       dtype=jax.numpy.int64),
        )

        def l_greaterThan_0(a: DynamicHubris):
            return (a['l'] > 0)[0]  # Extract scalar boolean

        def countFoldings(a: DynamicHubris):
            def noChange(a: DynamicHubris):
                return a
            def findFolds(a: DynamicHubris):

                def increment_foldings(a: DynamicHubris):
                    a['foldingsSubtotal'] += leavesTotal
                    return a

                def findGaps(a: DynamicHubris):
                    class DynamicFindGaps(DynamicHubris):
                        dd: jax.Array
                        gg: jax.Array

                    def countGaps(dimension1ndex: int, aFindGaps: DynamicFindGaps):
                        def ddUnconstrained(aFindGaps: DynamicFindGaps):
                            aFindGaps['dd'] += 1
                            return aFindGaps

                        def check_l_to_m(aFindGaps: DynamicFindGaps):
                            class DynamicCountGaps(DynamicFindGaps):
                                m: jax.Array

                            def m_notEqual_l(aCountGaps: DynamicCountGaps):
                                return (aCountGaps['m'] != aCountGaps['l'])[0]  # Extract scalar boolean

                            def smurfGapSmurf(aCountGaps: DynamicCountGaps):
                                def noChangeForYou(aCountGaps: DynamicCountGaps):
                                    return aCountGaps
                                def yourTaskDivision(aCountGaps: DynamicCountGaps):
                                    count, m = aCountGaps['count'], aCountGaps['m']
                                    aCountGaps['gap'] = aCountGaps['gap'].at[aCountGaps['gg']].set(m[0])
                                    aCountGaps['gg'] += jax.numpy.where((count[m] == 0), 1, 0)
                                    aCountGaps['count'] = aCountGaps['count'].at[m].set(count[m] + 1)
                                    return aCountGaps
                                if_yourTaskDivision = ((taskDivisions == 0) | 
                                                     (aCountGaps['l'] != taskDivisions) | 
                                                     ((aCountGaps['m'] % taskDivisions) == taskIndex))[0]
                                aCountGaps = jax.lax.cond(if_yourTaskDivision, yourTaskDivision, noChangeForYou, aCountGaps)
                                aCountGaps['m'] = D[dimension1ndex, aCountGaps['l'], aCountGaps['B'][aCountGaps['m']]]
                                return aCountGaps

                            aCountGaps: DynamicCountGaps = {
                                **aFindGaps,
                                'm': D[dimension1ndex, aFindGaps['l'], aFindGaps['l']]
                            }
                            aCountGaps = jax.lax.while_loop(m_notEqual_l, smurfGapSmurf, aCountGaps)

                            for keyName in aFindGaps.keys():
                                aFindGaps[keyName] = aCountGaps[keyName]
                            return aFindGaps

                        connectionGraphPointingAtSelf = (D[dimension1ndex, aFindGaps['l'], aFindGaps['l']] == aFindGaps['l'])[0]  # Extract scalar boolean
                        aFindGaps = jax.lax.cond(connectionGraphPointingAtSelf, ddUnconstrained, check_l_to_m, aFindGaps)
                        return aFindGaps

                    aFindGaps: DynamicFindGaps = {
                        **a,
                        'dd': jax.numpy.zeros(1, dtype=jax.numpy.int64),
                        'gg': a['gapter'][a['l'] - 1],
                        'g': a['gapter'][a['l'] - 1],
                    }
                    aFindGaps = jax.lax.fori_loop(1, dimensionsTotal + 1, countGaps, aFindGaps)

                    def allTiedUp(aFindGaps: DynamicFindGaps):
                        return aFindGaps

                    def unconstrainedLeaf(aFindGaps: DynamicFindGaps):
                        def for_m_in_range_l(m: int, aFindGaps: DynamicFindGaps):
                            aFindGaps['gap'] = aFindGaps['gap'].at[aFindGaps['gg']].set(m)
                            aFindGaps['gg'] += 1
                            return aFindGaps

                        aFindGaps = jax.lax.fori_loop(0, aFindGaps['l'][0], for_m_in_range_l, aFindGaps)
                        return aFindGaps

                    dd_equals_dimensionsTotal = (aFindGaps['dd'] == dimensionsTotal)[0]  # Extract scalar boolean
                    aFindGaps = jax.lax.cond(dd_equals_dimensionsTotal, unconstrainedLeaf, allTiedUp, aFindGaps)

                    def filterCommonGaps(j: int, aFindGaps: DynamicFindGaps):
                        aFindGaps['gap'] = aFindGaps['gap'].at[aFindGaps['g']].set(aFindGaps['gap'][j])
                        if_count_index_gap_index_j_equal_dimensionsTotal_minus_dd_and_venus_is_in_retrograde = aFindGaps['count'][aFindGaps['gap'][j]] == dimensionsTotal - aFindGaps['dd']
                        aFindGaps['g'] += jax.numpy.where(if_count_index_gap_index_j_equal_dimensionsTotal_minus_dd_and_venus_is_in_retrograde, 1, 0)
                        aFindGaps['count'] = aFindGaps['count'].at[aFindGaps['gap'][j]].set(0)
                        return aFindGaps

                    aFindGaps = jax.lax.fori_loop(aFindGaps['g'][0], aFindGaps['gg'][0], filterCommonGaps, aFindGaps)

                    for keyName in a.keys():
                        a[keyName] = aFindGaps[keyName]
                    return a

                l_LT_leavesTotal = (a['l'] > leavesTotal)[0]  # Extract scalar boolean
                a = jax.lax.cond(l_LT_leavesTotal, increment_foldings, findGaps, a)
                return a

            l_LTE_1_or_B_index_0_is_1 = ((a['l'] <= 1) | (a['B'] == 1))[0]  # Extract scalar boolean
            a = jax.lax.cond(l_LTE_1_or_B_index_0_is_1, findFolds, noChange, a)

            def l_GT_0_and_g_is_gapter_index_lMinus1(a: DynamicHubris):
                return ((a['l'] > 0) & (a['g'] == a['gapter'][a['l'] - 1]))[0]

            def backtrack(a: DynamicHubris):
                A = a['A']
                B = a['B']
                l = a['l']

                l -= 1

                A_index_l = A[l]
                B_index_l = B[l]
                B = B.at[A_index_l].set(B_index_l)
                A = A.at[B_index_l].set(A_index_l)

                a['A'] = A
                a['B'] = B
                a['l'] = l
                return a

            a = jax.lax.while_loop(l_GT_0_and_g_is_gapter_index_lMinus1, backtrack, a)

            def move_to_next_leaf(a: DynamicHubris):
                A = a['A']
                B = a['B']
                g = a['g']
                gap = a['gap']
                gapter = a['gapter']
                l = a['l']

                g -= 1

                A = A.at[l].set(gap[g])
                B = B.at[l].set(B[A[l]])
                B = B.at[A[l]].set(l)
                A = A.at[B[l]].set(l)
                gapter = gapter.at[l].set(g)
                l += 1

                a['A'] = A
                a['B'] = B
                a['g'] = g
                a['gap'] = gap
                a['gapter'] = gapter
                a['l'] = l
                return a

            a = jax.lax.cond(a['l'][0] > 0, move_to_next_leaf, noChange, a)
            return a

        dynamicHubris_output_val = jax.lax.while_loop(l_greaterThan_0, countFoldings, dynamicHubris_init_val)
        return dynamicHubris_output_val['foldingsSubtotal']

    arrayFoldingsSubtotals: jax.Array = jax.vmap(hubris)(arrayIndicesTask)

    return int(arrayFoldingsSubtotals.sum())
