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
import inspect
"""ideas:
- revisit optimizing dtype; except foldingsSubtotal, int8 is (probably) large enough
- revisit mapFoldingPathDivisions; n.b., only tried with `+= 1` not `+= leavesTotal`
"""
class Hubris(TypedDict):
    A: jax.Array
    B: jax.Array
    count: jax.Array
    foldingsSubtotal: jax.Array
    g: jax.Array
    gap: jax.Array
    gapter: jax.Array
    l: jax.Array

# reHubris = ['A', 'B', 'count', 'foldingsSubtotal', 'g', 'gap', 'gapter', 'l'] # not DRY: for troubleshooting

class HubHubrisris(TypedDict):
    """a, dd, gg"""
    a: Hubris
    dd: jax.Array
    gg: jax.Array

class HubHubHubrisrisris(TypedDict):
    """aHubHubrisris, m"""
    aHubHubrisris: HubHubrisris
    m: jax.Array

def spoon(taskDivisions: jax.Array, arrayIndicesTask: jax.Array, leavesTotal: jax.Array, dimensionsTotal: jax.Array, D: jax.Array):

    hubris_init_val =  Hubris(
        A                = jax.numpy.zeros(leavesTotal + 1,       dtype=jax.numpy.int64),
        B                = jax.numpy.zeros(leavesTotal + 1,       dtype=jax.numpy.int64),
        count            = jax.numpy.zeros(leavesTotal + 1,       dtype=jax.numpy.int64),
        foldingsSubtotal = jax.numpy.array(0, dtype=jax.numpy.int64),  
        g                = jax.numpy.array(0, dtype=jax.numpy.int64),  
        gap              = jax.numpy.zeros((leavesTotal **2) + 1, dtype=jax.numpy.int64),
        gapter           = jax.numpy.zeros(leavesTotal + 1,       dtype=jax.numpy.int64),
        l                = jax.numpy.array(1, dtype=jax.numpy.int64),  
    )

    @jax.jit
    def hubris(taskIndex: jax.Array):

        def l_greaterThan_0(a99: Hubris):
            return a99['l'] > 0  

        def countFoldings(a102: Hubris):
            def noChange(a103: Hubris):
                return a103
            def findFolds(a105: Hubris):
                def increment_foldings(a106: Hubris):
                    a107: Hubris = {'A': a106['A'], 'B': a106['B'], 'count': a106['count'], 'foldingsSubtotal': a106['foldingsSubtotal'] + leavesTotal, 'g': a106['g'], 'gap': a106['gap'], 'gapter': a106['gapter'], 'l': a106['l']}
                    jax.debug.print("increment_foldings: foldingsSubtotal={foldingsSubtotal}", foldingsSubtotal=a107['foldingsSubtotal'])
                    return a107

                def findGaps(a112: Hubris):
                    def countGaps(dimension1ndex: int, aHubHubrisris113: HubHubrisris):
                        def ddUnconstrained(aHubHubrisris114: HubHubrisris):
                            aHubHubrisris115: HubHubrisris = {'dd': aHubHubrisris114['dd'] + 1, 'a': aHubHubrisris114['a'], 'gg': aHubHubrisris114['gg']}
                            return aHubHubrisris115

                        def check_l_to_m(aHubHubrisris118: HubHubrisris):
                            def m_notEqual_l(aCountGaps119: HubHubHubrisrisris):
                                return aCountGaps119['m'] != aCountGaps119['aHubHubrisris']['a']['l']

                            def smurfGapSmurf(aCountGaps122: HubHubHubrisrisris):
                                def noChangeForYou(aCountGaps123: HubHubHubrisrisris):
                                    return aCountGaps123
                                def yourTaskDivision(aCountGaps125: HubHubHubrisrisris):
                                    count126 = aCountGaps125['aHubHubrisris']['a']['count']
                                    gap127 = aCountGaps125['aHubHubrisris']['a']['gap']
                                    gg128 = aCountGaps125['aHubHubrisris']['gg']
                                    m129 = aCountGaps125['m']

                                    gap127 = gap127.at[gg128].set(m129)
                                    gg128 += jax.numpy.where((count126[m129] == 0), 1, 0)
                                    count126 = count126.at[m129].set(count126[m129] + 1)

                                    a134: Hubris = {'A': aCountGaps125['aHubHubrisris']['a']['A'], 'B': aCountGaps125['aHubHubrisris']['a']['B'],
                                                    'count': count126,
                                                    'foldingsSubtotal': aCountGaps125['aHubHubrisris']['a']['foldingsSubtotal'], 'g': aCountGaps125['aHubHubrisris']['a']['g'], 
                                                    'gap': gap127, 'gapter': aCountGaps125['aHubHubrisris']['a']['gapter'], 'l': aCountGaps125['aHubHubrisris']['a']['l']}
                                    aHubHubrisris138: HubHubrisris = {'a': a134, 'dd': aCountGaps125['aHubHubrisris']['dd'], 'gg': gg128}
                                    aCountGaps139: HubHubHubrisrisris = {'aHubHubrisris': aHubHubrisris138, 'm': m129}

                                    return aCountGaps139

                                if_yourTaskDivision = (taskDivisions == 0) | (aCountGaps122['aHubHubrisris']['a']['l'] != taskDivisions) | ((aCountGaps122['m'] % taskDivisions) == taskIndex)
                                aCountGaps144: HubHubHubrisrisris = jax.lax.cond(if_yourTaskDivision, yourTaskDivision, noChangeForYou, aCountGaps122)
                                aCountGaps145: HubHubHubrisrisris = {
                                    'm':  D[dimension1ndex, aCountGaps144['aHubHubrisris']['a']['l'], aCountGaps144['aHubHubrisris']['a']['B'][aCountGaps144['m']]],
                                    'aHubHubrisris': aCountGaps144['aHubHubrisris']}
                                return aCountGaps145

                            aCountGaps150: HubHubHubrisrisris = {
                                'aHubHubrisris': aHubHubrisris118,
                                'm': D[dimension1ndex, aHubHubrisris118['a']['l'], aHubHubrisris118['a']['l']]
                            }
                            aCountGaps154: HubHubHubrisrisris = jax.lax.while_loop(m_notEqual_l, smurfGapSmurf, aCountGaps150)
                            
                            deconstructed156: HubHubrisris = aCountGaps154['aHubHubrisris']

                            return deconstructed156

                        connectionGraphPointingAtSelf = D[dimension1ndex, aHubHubrisris113['a']['l'], aHubHubrisris113['a']['l']] == aHubHubrisris113['a']['l']
                        aHubHubrisris161: HubHubrisris = jax.lax.cond(connectionGraphPointingAtSelf, ddUnconstrained, check_l_to_m, aHubHubrisris113)
                        return aHubHubrisris161

                    dd164 = jax.numpy.array(0, dtype=jax.numpy.int64)
                    gg165 = a112['gapter'][a112['l'] - 1]
                    a116: Hubris = {'A': a112['A'], 'B': a112['B'], 'count': a112['count'], 'foldingsSubtotal': a112['foldingsSubtotal'],
                                    'g': gg165, 'gap': a112['gap'], 'gapter': a112['gapter'], 'l': a112['l']}

                    aHubHubrisris169: HubHubrisris = {'a': a116, 'dd': dd164, 'gg': gg165}
                    aHubHubrisris170: HubHubrisris = jax.lax.fori_loop(1, dimensionsTotal + 1, countGaps, aHubHubrisris169)

                    def stitchedUp(aHubHubrisris172: HubHubrisris):
                        return aHubHubrisris172

                    def unconstrainedLeaf(aHubHubrisris175: HubHubrisris):
                        def for_m_in_range_l(m176: int, aHubHubrisris176: HubHubrisris):
                            gap178 = aHubHubrisris176['a']['gap']
                            gap178 = gap178.at[aHubHubrisris176['a']['g']].set(m176)
                            a179: Hubris = {'A': aHubHubrisris176['a']['A'], 'B': aHubHubrisris176['a']['B'], 'count': aHubHubrisris176['a']['count'],
                                            'foldingsSubtotal': aHubHubrisris176['a']['foldingsSubtotal'], 'g': aHubHubrisris176['a']['g'], 'gap': gap178,
                                            'gapter': aHubHubrisris176['a']['gapter'], 'l': aHubHubrisris176['a']['l']}
                            aHubHubrisris182: HubHubrisris = {'a': a179, 'dd': aHubHubrisris176['dd'], 'gg': aHubHubrisris176['gg'] + 1}
                            return aHubHubrisris182

                        aHubHubrisris185: HubHubrisris = jax.lax.fori_loop(0, aHubHubrisris175['a']['l'], for_m_in_range_l, aHubHubrisris175)
                        return aHubHubrisris185

                    dd_equals_dimensionsTotal = aHubHubrisris170['dd'] == dimensionsTotal
                    aHubHubrisris189: HubHubrisris = jax.lax.cond(dd_equals_dimensionsTotal, unconstrainedLeaf, stitchedUp, aHubHubrisris170)

                    def filterCommonGaps(j: int, aHubHubrisris191: HubHubrisris):
                        gap192 = aHubHubrisris191['a']['gap']
                        count193 = aHubHubrisris191['a']['count']
                        g194 = aHubHubrisris191['a']['g']
                        gap192 = gap192.at[g194].set(gap192[j])
                        if196 = count193[gap192[j]] == dimensionsTotal - aHubHubrisris191['dd']
                        g197 = g194 + jax.numpy.where(if196, 1, 0)
                        count193 = count193.at[gap192[j]].set(0)
                        a199: Hubris = {'A': aHubHubrisris191['a']['A'], 'B': aHubHubrisris191['a']['B'], 'count': count193,
                                        'foldingsSubtotal': aHubHubrisris191['a']['foldingsSubtotal'], 'g': g197, 'gap': gap192,
                                        'gapter': aHubHubrisris191['a']['gapter'], 'l': aHubHubrisris191['a']['l']}
                        aHubHubrisris202: HubHubrisris = {'a': a199, 'dd': aHubHubrisris191['dd'], 'gg': aHubHubrisris191['gg']}
                        return aHubHubrisris202

                    aHubHubrisris205: HubHubrisris = jax.lax.fori_loop(aHubHubrisris189['a']['g'], aHubHubrisris189['gg'], filterCommonGaps, aHubHubrisris189)

                    a207: Hubris = aHubHubrisris205['a']

                    return a207

                l_greaterThan_leavesTotal = a105['l'] > leavesTotal
                a212: Hubris = jax.lax.cond(l_greaterThan_leavesTotal, increment_foldings, findGaps, a105)
                return a212

            l_LTE_1_or_B_index_0_is_1 = (a102['l'] <= 1) | (a102['B'][0] == 1)
            a216: Hubris = jax.lax.cond(l_LTE_1_or_B_index_0_is_1, findFolds, noChange, a102)
            jax.debug.print("line no:{line} foldingsSubtotal={foldingsSubtotal}", line=inspect.currentframe().f_lineno, foldingsSubtotal=a216['foldingsSubtotal']) #type: ignore

            def l_GT_0_and_g_is_gapter_index_lMinus1(a219: Hubris):
                return (a219['l'] > 0) & (a219['g'] == a219['gapter'][a219['l'] - 1])

            def backtrack(a222: Hubris):
                A223 = a222['A']
                B224 = a222['B']
                l225 = a222['l'] - 1

                A_index_l227 = A223[l225]
                B_index_l228 = B224[l225]
                B224 = B224.at[A_index_l227].set(B_index_l228)
                A_index_l230 = A223[l225]
                B_index_l231 = B224[l225]
                A223 = A223.at[B_index_l231].set(A_index_l230)

                a234: Hubris = {'A': A223, 'B': B224, 'count': a222['count'], 'foldingsSubtotal': a222['foldingsSubtotal'], 'g': a222['g'], 'gap': a222['gap'], 'gapter': a222['gapter'], 'l': l225}
                return a234

            a237: Hubris = jax.lax.while_loop(l_GT_0_and_g_is_gapter_index_lMinus1, backtrack, a216)

            def move_to_next_leaf(a239: Hubris):
                A240 = a239['A']
                B241 = a239['B']
                g242 = a239['g'] - 1
                gap243 = a239['gap']
                gapter244 = a239['gapter']
                l245 = a239['l']

                A240 = A240.at[l245].set(gap243[g242])
                B241 = B241.at[l245].set(B241[A240[l245]])
                B241 = B241.at[A240[l245]].set(l245)
                A240 = A240.at[B241[l245]].set(l245)
                gapter244 = gapter244.at[l245].set(g242)
                l252 = l245 + 1

                a254: Hubris = {'A': A240, 'B': B241, 'count': a239['count'], 'foldingsSubtotal': a239['foldingsSubtotal'], 
                                'g': g242, 'gap': gap243, 'gapter': gapter244, 'l': l252}
                return a254

            a264: Hubris = jax.lax.cond(a237['l'] > 0, move_to_next_leaf, noChange, a237)

            # jax.debug.print("End of countFoldings: l={l}, g={g}, foldingsSubtotal={foldingsSubtotal}", l=a['l'], g=a['g'], foldingsSubtotal=a['foldingsSubtotal'])

            return a264

        hubris_output_val = jax.lax.while_loop(l_greaterThan_0, countFoldings, hubris_init_val)
        return hubris_output_val['foldingsSubtotal']

    arrayFoldingsSubtotals: jax.Array = jax.vmap(hubris)(arrayIndicesTask)

    return int(arrayFoldingsSubtotals.sum())
