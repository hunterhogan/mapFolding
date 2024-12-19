"""For troubleshooting, special identifiers:

- Where possible, identifiers match the working non-JAX version: https://github.com/hunterhogan/mapFolding/blob/68a1d7f900caa92916edaba6c2a6dcfd229b3c3a/mapFolding/lovelace.py
- `dynamicHubris_init_val`: parameter `init_val` in `jax.lax.while_loop`, "of type ``a``"
- `dynamicHubris_output_val`: return value of `jax.lax.while_loop`, "of type ``a``"
- `a`: within the while loop, the `dynamicHubris` "of type ``a``", which must be explicitly handled in JAX-style functions
- Sometimes, some conditional statements must be given an identifier, such as "cond_fun" in `jax.lax.while_loop`
    and the predicate in `jax.lax.cond`. The identifier attempts to represent the logic of the conditional
    statement. So `l > 0`, for example, could be `l_greaterThan_0`.

By adding line numbers to identifiers, they should all be unique.
"""
"""ideas:
- revisit optimizing dtype; except foldingsSubtotal, int8 is (probably) large enough
- revisit mapFoldingPathDivisions; n.b., only tried with `+= 1` not `+= leavesTotal`
"""
from typing import TypedDict
import jax
import inspect
from .pidVariables import integerSize

class Hubris(TypedDict):
    A: jax.Array
    B: jax.Array
    count: jax.Array
    foldingsSubtotal: jax.Array
    g: jax.Array
    gap: jax.Array
    gapter: jax.Array
    l: jax.Array

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

    hubris_init_val43 =  Hubris(
        A                = jax.numpy.zeros(leavesTotal + 1,       dtype=integerSize),
        B                = jax.numpy.zeros(leavesTotal + 1,       dtype=integerSize),
        count            = jax.numpy.zeros(leavesTotal + 1,       dtype=integerSize),
        foldingsSubtotal = jax.numpy.array(0, dtype=integerSize),  
        g                = jax.numpy.array(0, dtype=integerSize),  
        gap              = jax.numpy.zeros((leavesTotal **2) + 1, dtype=integerSize),
        gapter           = jax.numpy.zeros(leavesTotal + 1,       dtype=integerSize),
        l                = jax.numpy.array(1, dtype=integerSize),  
    )

    @jax.jit
    def hubris(taskIndex55: jax.Array):

        def l_greaterThan_0(q57: Hubris):
            return q57['l'] > 0  

        def countFoldings(w60: Hubris):
            def noChange(e61: Hubris):
                
                return e61
            def findFolds(r63: Hubris):
                def increment_foldings(t64: Hubris):
                    y65: Hubris = {'A': t64['A'], 'B': t64['B'], 'count': t64['count'], 'foldingsSubtotal': t64['foldingsSubtotal'] + leavesTotal, 'g': t64['g'], 'gap': t64['gap'], 'gapter': t64['gapter'], 'l': t64['l']}
                    jax.debug.print("increment_foldings: foldingsSubtotal={foldingsSubtotal}", foldingsSubtotal=y65['foldingsSubtotal'])
                    return y65

                def findGaps(u69: Hubris):
                    def countGaps(dimension1ndex70: int, i70: HubHubrisris):
                        def ddUnconstrained(o71: HubHubrisris):
                            p72: HubHubrisris = {'dd': o71['dd'] + 1, 'a': o71['a'], 'gg': o71['gg']}
                            return p72

                        def check_l_to_m(s75: HubHubrisris):
                            def m_notEqual_l(d76: HubHubHubrisrisris):
                                return d76['m'] != d76['aHubHubrisris']['a']['l']

                            def smurfGapSmurf(f79: HubHubHubrisrisris):
                                def noChangeForYou(kk80: HubHubHubrisrisris):
                                    return kk80
                                def yourTaskDivision(h82: HubHubHubrisrisris):
                                    count_83 = h82['aHubHubrisris']['a']['count']
                                    gap_84 = h82['aHubHubrisris']['a']['gap']
                                    gg_85 = h82['aHubHubrisris']['gg']
                                    m_86 = h82['m']

                                    gap_84 = gap_84.at[gg_85].set(m_86)
                                    gg_85 += jax.numpy.where((count_83[m_86] == 0), jax.numpy.array(1, dtype=integerSize), jax.numpy.array(0, dtype=integerSize))
                                    # count_83 = count_83.at[m_86].set(count_83[m_86] + 1)
                                    count_83 = count_83.at[m_86].add(1)

                                    k92: Hubris = {'A': h82['aHubHubrisris']['a']['A'], 'B': h82['aHubHubrisris']['a']['B'],
                                                    'count': count_83,
                                                    'foldingsSubtotal': h82['aHubHubrisris']['a']['foldingsSubtotal'], 'g': h82['aHubHubrisris']['a']['g'], 
                                                    'gap': gap_84, 'gapter': h82['aHubHubrisris']['a']['gapter'], 'l': h82['aHubHubrisris']['a']['l']}
                                    z96: HubHubrisris = {'a': k92, 'dd': h82['aHubHubrisris']['dd'], 'gg': gg_85}
                                    x97: HubHubHubrisrisris = {'aHubHubrisris': z96, 'm': m_86}

                                    return x97

                                if_yourTaskDivision = (taskDivisions == 0) | (f79['aHubHubrisris']['a']['l'] != taskDivisions) | ((f79['m'] % taskDivisions) == taskIndex55)
                                c102: HubHubHubrisrisris = jax.lax.cond(if_yourTaskDivision, yourTaskDivision, noChangeForYou, f79)
                                v103: HubHubHubrisrisris = {
                                    'm':  D[dimension1ndex70, c102['aHubHubrisris']['a']['l'], c102['aHubHubrisris']['a']['B'][c102['m']]],
                                    'aHubHubrisris': c102['aHubHubrisris']}
                                return v103

                            b108: HubHubHubrisrisris = {
                                'aHubHubrisris': s75,
                                'm': D[dimension1ndex70, s75['a']['l'], s75['a']['l']]
                            }
                            n112: HubHubHubrisrisris = jax.lax.while_loop(m_notEqual_l, smurfGapSmurf, b108)
                            
                            qq114: HubHubrisris = n112['aHubHubrisris']

                            return qq114

                        connectionGraphPointingAtSelf = D[dimension1ndex70, i70['a']['l'], i70['a']['l']] == i70['a']['l']
                        ww119: HubHubrisris = jax.lax.cond(connectionGraphPointingAtSelf, ddUnconstrained, check_l_to_m, i70)
                        return ww119

                    dd122 = jax.numpy.array(0, dtype=integerSize)
                    gg123 = u69['gapter'][u69['l'] - 1]
                    ee124: Hubris = {'A': u69['A'], 'B': u69['B'], 'count': u69['count'], 'foldingsSubtotal': u69['foldingsSubtotal'],
                                    'g': gg123, 'gap': u69['gap'], 'gapter': u69['gapter'], 'l': u69['l']}

                    rr127: HubHubrisris = {'a': ee124, 'dd': dd122, 'gg': gg123}
                    tt128: HubHubrisris = jax.lax.fori_loop(1, dimensionsTotal + 1, countGaps, rr127)

                    def stitchedUp(yy130: HubHubrisris):
                        return yy130

                    def unconstrainedLeaf(uu133: HubHubrisris):
                        def for_m_in_range_l(m134: int, ii134: HubHubrisris):
                            gap135 = ii134['a']['gap']
                            gap135 = gap135.at[ii134['a']['g']].set(m134)
                            oo137: Hubris = {'A': ii134['a']['A'], 'B': ii134['a']['B'], 'count': ii134['a']['count'],
                                            'foldingsSubtotal': ii134['a']['foldingsSubtotal'], 'g': ii134['a']['g'], 'gap': gap135,
                                            'gapter': ii134['a']['gapter'], 'l': ii134['a']['l']}
                            pp140: HubHubrisris = {'a': oo137, 'dd': ii134['dd'], 'gg': ii134['gg'] + 1}
                            return pp140

                        ss143: HubHubrisris = jax.lax.fori_loop(0, uu133['a']['l'], for_m_in_range_l, uu133)
                        return ss143

                    dd_equals_dimensionsTotal = tt128['dd'] == dimensionsTotal
                    ff147: HubHubrisris = jax.lax.cond(dd_equals_dimensionsTotal, unconstrainedLeaf, stitchedUp, tt128)

                    def filterCommonGaps(j149: int, hh149: HubHubrisris):
                        gap150 = hh149['a']['gap']
                        count151 = hh149['a']['count']
                        g152 = hh149['a']['g']
                        gap150 = gap150.at[g152].set(gap150[j149])
                        if154 = count151[gap150[j149]] == dimensionsTotal - hh149['dd']
                        g155 = g152 + jax.numpy.where(if154, 1, 0)
                        count151 = count151.at[gap150[j149]].set(0)
                        ll157: Hubris = {'A': hh149['a']['A'], 'B': hh149['a']['B'], 'count': count151,
                                        'foldingsSubtotal': hh149['a']['foldingsSubtotal'], 'g': g155, 'gap': gap150,
                                        'gapter': hh149['a']['gapter'], 'l': hh149['a']['l']}
                        zz160: HubHubrisris = {'a': ll157, 'dd': hh149['dd'], 'gg': hh149['gg']}
                        return zz160

                    xx163: HubHubrisris = jax.lax.fori_loop(ff147['a']['g'], ff147['gg'], filterCommonGaps, ff147)

                    cc165: Hubris = xx163['a']

                    return cc165

                l_greaterThan_leavesTotal = r63['l'] > leavesTotal
                vv170: Hubris = jax.lax.cond(l_greaterThan_leavesTotal, increment_foldings, findGaps, r63)
                return vv170

            l_LTE_1_or_B_index_0_is_1 = (w60['l'] <= 1) | (w60['B'][0] == 1)
            bb174: Hubris = jax.lax.cond(l_LTE_1_or_B_index_0_is_1, findFolds, noChange, w60)
            jax.debug.print("line no:{line} foldingsSubtotal={foldingsSubtotal}", line=inspect.currentframe().f_lineno, foldingsSubtotal=bb174['foldingsSubtotal']) #type: ignore

            def l_GT_0_and_g_is_gapter_index_lMinus1(nn177: Hubris):
                return (nn177['l'] > 0) & (nn177['g'] == nn177['gapter'][nn177['l'] - 1])

            def backtrack(mm180: Hubris):
                A181 = mm180['A']
                B182 = mm180['B']
                l183 = mm180['l'] - 1

                A_index_l185 = A181[l183]
                B_index_l186 = B182[l183]
                B182 = B182.at[A_index_l185].set(B_index_l186)
                A_index_l188 = A181[l183]
                B_index_l189 = B182[l183]
                A181 = A181.at[B_index_l189].set(A_index_l188)

                qqq192: Hubris = {'A': A181, 'B': B182, 'count': mm180['count'], 'foldingsSubtotal': mm180['foldingsSubtotal'], 'g': mm180['g'], 'gap': mm180['gap'], 'gapter': mm180['gapter'], 'l': l183}
                return qqq192

            www195: Hubris = jax.lax.while_loop(l_GT_0_and_g_is_gapter_index_lMinus1, backtrack, bb174)

            def move_to_next_leaf(eee197: Hubris):
                A198 = eee197['A']
                B199 = eee197['B']
                g200 = eee197['g'] - 1
                gap201 = eee197['gap']
                gapter202 = eee197['gapter']
                l203 = eee197['l']

                A198 = A198.at[l203].set(gap201[g200])
                B199 = B199.at[l203].set(B199[A198[l203]])
                B199 = B199.at[A198[l203]].set(l203)
                A198 = A198.at[B199[l203]].set(l203)
                gapter202 = gapter202.at[l203].set(g200)
                l210 = l203 + 1

                rrr212: Hubris = {'A': A198, 'B': B199, 'count': eee197['count'], 'foldingsSubtotal': eee197['foldingsSubtotal'], 
                                'g': g200, 'gap': gap201, 'gapter': gapter202, 'l': l210}
                return rrr212

            ttt216: Hubris = jax.lax.cond(www195['l'] > 0, move_to_next_leaf, noChange, www195)
            return ttt216

        hubris_output_val219 = jax.lax.while_loop(l_greaterThan_0, countFoldings, hubris_init_val43)
        return hubris_output_val219['foldingsSubtotal']

    arrayFoldingsSubtotals: jax.Array = jax.vmap(hubris)(arrayIndicesTask)

    return int(arrayFoldingsSubtotals.sum())
