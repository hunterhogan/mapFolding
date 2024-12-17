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

def spoon(taskDivisions: jax.Array, arrayIndicesTask: jax.Array, leavesTotal: jax.Array, dimensionsTotal: jax.Array, D: jax.Array) -> int:

    class DynamicHubris(TypedDict):
        A: jax.Array
        B: jax.Array
        count: jax.Array
        foldingsSubtotal: jax.Array
        g: jax.Array
        gap: jax.Array
        gapter: jax.Array
        l: jax.Array

    def hubris(taskIndex: jax.Array) -> jax.Array:
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

        def l_greaterThan_0(a: DynamicHubris) -> jax.Array:
            return a['l'][0] > 0

        def countFoldings(a: DynamicHubris):
            def noChange(a: DynamicHubris) -> DynamicHubris:
                return a
            def findFolds(a: DynamicHubris):

                def increment_foldings(a: DynamicHubris) -> DynamicHubris:
                    a['foldingsSubtotal'] += leavesTotal
                    return a

                def findGaps(a: DynamicHubris) -> DynamicHubris:
                    class DynamicFindGaps(DynamicHubris):
                        dd: jax.Array
                        gg: jax.Array

                    def countGaps(dimension1ndex: int, aFindGaps: DynamicFindGaps) -> DynamicFindGaps:
                        def ddUnconstrained(aFindGaps: DynamicFindGaps) -> DynamicFindGaps:
                            aFindGaps['dd'] += 1
                            return aFindGaps

                        def check_l_to_m(aFindGaps: DynamicFindGaps) -> DynamicFindGaps:
                            class DynamicCountGaps(DynamicFindGaps):
                                m: jax.Array

                            def m_notEqual_l(aCountGaps: DynamicCountGaps) -> jax.Array:
                                return aCountGaps['m'][0] != aCountGaps['l'][0]

                            def smurfGapSmurf(aCountGaps: DynamicCountGaps) -> DynamicCountGaps:
                                def noChangeForYou(aCountGaps: DynamicCountGaps) -> DynamicCountGaps:
                                    return aCountGaps
                                def yourTaskDivision(aCountGaps: DynamicCountGaps) -> DynamicCountGaps:
                                    count, m = aCountGaps['count'], aCountGaps['m']
                                    aCountGaps['gap'] = aCountGaps['gap'].at[aCountGaps['gg']].set(m[0])
                                    aCountGaps['gg'] += jax.numpy.where((count[m] == 0), 1, 0)
                                    aCountGaps['count'] = aCountGaps['count'].at[m].set(count[m] + 1)
                                    return aCountGaps
                                if_yourTaskDivision = (taskDivisions == 0) | (aCountGaps['l'][0] != taskDivisions) | ((aCountGaps['m'][0] % taskDivisions) == taskIndex)
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

                        connectionGraphPointingAtSelf = D[dimension1ndex, aFindGaps['l'][0], aFindGaps['l'][0]] == aFindGaps['l'][0]
                        aFindGaps = jax.lax.cond(connectionGraphPointingAtSelf, ddUnconstrained, check_l_to_m, aFindGaps)
                        return aFindGaps

                    aFindGaps: DynamicFindGaps = {
                        **a,
                        'dd': jax.numpy.zeros(1, dtype=jax.numpy.int64),
                        'gg': a['gapter'][a['l'] - 1],
                        'g': a['gapter'][a['l'] - 1],
                    }
                    aFindGaps = jax.lax.fori_loop(1, dimensionsTotal + 1, countGaps, aFindGaps)

                    def allTiedUp(aFindGaps: DynamicFindGaps) -> DynamicFindGaps:
                        return aFindGaps

                    def unconstrainedLeaf(aFindGaps: DynamicFindGaps) -> DynamicFindGaps:
                        def for_m_in_range_l(m, aFindGaps: DynamicFindGaps) -> DynamicFindGaps:
                            aFindGaps['gap'] = aFindGaps['gap'].at[aFindGaps['gg']].set(m)
                            aFindGaps['gg'] += 1
                            return aFindGaps

                        aFindGaps = jax.lax.fori_loop(0, aFindGaps['l'], for_m_in_range_l, aFindGaps)
                        return aFindGaps

                    dd_equals_dimensionsTotal = aFindGaps['dd'][0] == dimensionsTotal
                    aFindGaps = jax.lax.cond(dd_equals_dimensionsTotal, unconstrainedLeaf, allTiedUp, aFindGaps)

                    def filterCommonGaps(j, aFindGaps: DynamicFindGaps) -> DynamicFindGaps:
                        gap = aFindGaps['gap']
                        g = aFindGaps['g']
                        count = aFindGaps['count']
                        dd = aFindGaps['dd']
                        gap = gap.at[g].set(gap[j])
                        condition_count = count[gap[j]] == dimensionsTotal - dd
                        g += jax.numpy.where(condition_count, 1, 0)
                        count = count.at[gap[j]].set(0)
                        aFindGaps['gap'] = gap
                        aFindGaps['g'] = g
                        aFindGaps['count'] = count
                        return aFindGaps

                    aFindGaps = jax.lax.fori_loop(aFindGaps['g'], aFindGaps['gg'], filterCommonGaps, aFindGaps)

                    for keyName in a.keys():
                        a[keyName] = aFindGaps[keyName]
                    return a

                l_LT_leavesTotal = a['l'][0] > leavesTotal
                a = jax.lax.cond(l_LT_leavesTotal, increment_foldings, findGaps, a)
                return a

            l_LTE_1_or_B_index_0_is_1 = (a['l'][0] <= 1) | (a['B'][0] == 1)
            a = jax.lax.cond(l_LTE_1_or_B_index_0_is_1, findFolds, noChange, a)

            def l_GT_0_and_g_is_gapter_index_lMinus1(a: DynamicHubris) -> jax.Array:
                return ((a['l'][0] > 0) & (a['g'][0] == a['gapter'][a['l'] - 1][0]))[0]  # Extract scalar boolean

            def backtrack(a: DynamicHubris) -> DynamicHubris:
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

            def move_to_next_leaf(a: DynamicHubris) -> DynamicHubris:
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
