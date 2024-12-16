import jax
import jax.numpy as jnp
import functools

def spoon(computationDivisions: int, arrayIndicesComputation: jax.Array, countLeavesTotal: int, countDimensionsTotal: int, arrayD: jax.Array) -> int:
    """Computes total 'foldings' using JAX and lax.while_loop.

    Parameters:
        computationDivisions: Number of divisions for parallel computation.
        arrayIndicesComputation: Array of computation indices.
        countLeavesTotal: Total number of leaves.
        countDimensionsTotal: Total number of dimensions.
        arrayD: Precomputed array for leaf connections.
    """

    def hubris(computationIndex, computationDivisions, countLeavesTotal, countDimensionsTotal, arrayD): 
        totalFoldings = jnp.array(0, dtype=jnp.int64)
        indexLeafActive = jnp.array(1, dtype=jnp.int64)
        indexGapActive = jnp.array(0, dtype=jnp.int64)
        arrayLeafAbove = jnp.zeros(countLeavesTotal + 1, dtype=jnp.int64)
        arrayLeafBelow = jnp.zeros(countLeavesTotal + 1, dtype=jnp.int64)
        arrayCount = jnp.zeros(countLeavesTotal + 1, dtype=jnp.int64)
        arrayGapter = jnp.zeros(countLeavesTotal + 1, dtype=jnp.int64)
        arrayGap = jnp.zeros((countLeavesTotal * countLeavesTotal) + 1, dtype=jnp.int64)
        dimensionCountUnconstrained = jnp.array(0, dtype=jnp.int64)

        carry = {
            'arrayLeafAbove': arrayLeafAbove,
            'arrayLeafBelow': arrayLeafBelow,
            'arrayCount': arrayCount,
            'arrayGapter': arrayGapter,
            'arrayGap': arrayGap,
            'totalFoldings': totalFoldings,
            'indexGapActive': indexGapActive,
            'indexLeafActive': indexLeafActive,
            'computationDivisions': computationDivisions,
            'computationIndex': computationIndex,
            'arrayD': arrayD,
            'countLeavesTotal': countLeavesTotal,
            'countDimensionsTotal': countDimensionsTotal,
            'dimensionCountUnconstrained': dimensionCountUnconstrained  
        }

        def activeLeafGT0(carry):
            return carry['indexLeafActive'] > 0

        def countFoldings(carry):
            indexLeafActive = carry['indexLeafActive']
            indexGapActive = carry['indexGapActive']
            arrayLeafAbove = carry['arrayLeafAbove']
            arrayLeafBelow = carry['arrayLeafBelow']
            arrayCount = carry['arrayCount']
            arrayGapter = carry['arrayGapter']
            arrayGap = carry['arrayGap']
            totalFoldings = carry['totalFoldings']
            computationDivisions = carry['computationDivisions']
            computationIndex = carry['computationIndex']
            arrayD = carry['arrayD']
            countLeavesTotal = carry['countLeavesTotal']
            countDimensionsTotal = carry['countDimensionsTotal']

            carry['dimensionCountUnconstrained'] = 0 

            ifFirstLeaf = (indexLeafActive <= 1) | (carry['arrayLeafBelow'][0] == 1)
            jax.debug.print("ifFirstLeaf: {} {} {} {}", carry['arrayLeafBelow'], carry['arrayLeafAbove'], carry['arrayCount'], carry['arrayGapter'])

            def true_branch(carry):
                indexLeafActive = carry['indexLeafActive']
                condition_overflow = indexLeafActive > carry['countLeavesTotal']

                def increment_foldings(carry):
                    totalFoldings = carry['totalFoldings'] + carry['countLeavesTotal']
                    carry = {**carry, 'totalFoldings': totalFoldings, 'dimensionCountUnconstrained': jnp.array(0, dtype=jnp.int64)}
                    return carry

                def process_leaves(carry):
                    dimensionCountUnconstrained = carry['dimensionCountUnconstrained']
                    indexGapper = carry['arrayGapter'][indexLeafActive - 1]
                    indexGapActive = indexGapper
                    arrayGap = carry['arrayGap']
                    arrayCount = carry['arrayCount']

                    def fori_body(indexDimension, val):
                        dimensionCountUnconstrained, indexGapActive, indexGapper, arrayGap, arrayCount = val
                        conditionDimension = carry['arrayD'][indexDimension, indexLeafActive, indexLeafActive] == indexLeafActive
                        dimensionCountUnconstrained += jnp.where(conditionDimension, 1, 0)

                        indexM = carry['arrayD'][indexDimension, indexLeafActive, indexLeafActive]

                        def while_cond(state):
                            indexM, indexGapper, arrayGap, arrayCount = state
                            return indexM != indexLeafActive

                        def while_body(state):
                            indexM, indexGapper, arrayGap, arrayCount = state
                            conditionIndexM = (
                                (carry['computationDivisions'] == 0) |
                                (indexLeafActive != carry['computationDivisions']) |
                                ((indexM % carry['computationDivisions']) == carry['computationIndex'])
                            )
                            arrayGap = arrayGap.at[indexGapper].set(
                                jnp.where(conditionIndexM, indexM, arrayGap[indexGapper])
                            )
                            arrayCount = arrayCount.at[indexM].set(
                                arrayCount[indexM] + jnp.where(conditionIndexM, 1, 0)
                            )
                            increment = jnp.where(
                                conditionIndexM & (arrayCount[indexM] == 1), 1, 0
                            )
                            indexGapper += increment
                            indexM = carry['arrayD'][indexDimension, indexLeafActive, carry['arrayLeafBelow'][indexM]]
                            return (indexM, indexGapper, arrayGap, arrayCount)

                        stateInit = (indexM, indexGapper, arrayGap, arrayCount)
                        stateFinal = jax.lax.while_loop(while_cond, while_body, stateInit)
                        _, indexGapper, arrayGap, arrayCount = stateFinal

                        return (
                            dimensionCountUnconstrained,
                            indexGapActive,
                            indexGapper,
                            arrayGap,
                            arrayCount
                        )

                    valInit = (
                        dimensionCountUnconstrained,
                        indexGapActive,
                        indexGapper,
                        arrayGap,
                        arrayCount
                    )
                    valFinal = jax.lax.fori_loop(
                        1, carry['countDimensionsTotal'] + 1, fori_body, valInit
                    )
                    dimensionCountUnconstrained, indexGapActive, indexGapper, arrayGap, arrayCount = valFinal

                    condition_all_unconstrained = dimensionCountUnconstrained == carry['countDimensionsTotal']

                    def if_unconstrained(val):
                        arrayGap, indexGapper = val

                        def fori_m_body(m, val):
                            arrayGap, indexGapper = val
                            arrayGap = arrayGap.at[indexGapper].set(m)
                            indexGapper += 1
                            return (arrayGap, indexGapper)

                        val_m = (arrayGap, indexGapper)
                        val_m_final = jax.lax.fori_loop(0, indexLeafActive, fori_m_body, val_m)
                        arrayGap, indexGapper = val_m_final
                        return (arrayGap, indexGapper)

                    def if_constrained(val):
                        return val

                    arrayGap, indexGapper = jax.lax.cond(condition_all_unconstrained, if_unconstrained, if_constrained, (arrayGap, indexGapper))

                    def for_j_body(j, val):
                        arrayGap, indexGapActive, arrayCount = val
                        arrayGap = arrayGap.at[indexGapActive].set(arrayGap[j])
                        condition_count = arrayCount[arrayGap[j]] == carry['countDimensionsTotal'] - dimensionCountUnconstrained
                        indexGapActive += jnp.where(condition_count, 1, 0)
                        arrayCount = arrayCount.at[arrayGap[j]].set(0)
                        return (arrayGap, indexGapActive, arrayCount)

                    val_j_init = (arrayGap, indexGapActive, arrayCount)
                    val_j_final = jax.lax.fori_loop(indexGapActive, indexGapper, for_j_body, val_j_init)
                    arrayGap, indexGapActive, arrayCount = val_j_final

                    carry = {**carry,
                             'dimensionCountUnconstrained': dimensionCountUnconstrained,
                             'indexGapActive': indexGapActive,
                             'arrayGap': arrayGap,
                             'arrayCount': arrayCount}
                    return carry

                carry = jax.lax.cond(condition_overflow, increment_foldings, process_leaves, carry)
                return carry

            carry = jax.lax.cond(ifFirstLeaf, true_branch, lambda c: c, carry)

            def while_cond_inner(state):
                indexLeafActive, indexGapActive, arrayGapter, carry = state
                return (indexLeafActive > 0) & (indexGapActive == arrayGapter[indexLeafActive - 1])

            def while_body_inner(state):
                indexLeafActive, indexGapActive, arrayGapter, carry = state
                indexLeafActive -= 1
                arrayLeafBelow = carry['arrayLeafBelow']
                arrayLeafAbove = carry['arrayLeafAbove']
                indexAbove = arrayLeafAbove[indexLeafActive]
                indexBelow = arrayLeafBelow[indexLeafActive]

                arrayLeafBelow = arrayLeafBelow.at[indexAbove].set(indexBelow)
                arrayLeafAbove = arrayLeafAbove.at[indexBelow].set(indexAbove)

                carry = {**carry, 'arrayLeafAbove': arrayLeafAbove, 'arrayLeafBelow': arrayLeafBelow}
                return (indexLeafActive, indexGapActive, arrayGapter, carry)

            state_init = (indexLeafActive, indexGapActive, carry['arrayGapter'], carry)
            state_final = jax.lax.while_loop(while_cond_inner, while_body_inner, state_init)
            indexLeafActive, indexGapActive, arrayGapter, carry = state_final

            carry['indexLeafActive'] = indexLeafActive
            carry['indexGapActive'] = indexGapActive

            def true_branch_leaf(carry):
                indexGapActive = carry['indexGapActive'] - 1
                arrayGap = carry['arrayGap']
                arrayLeafAbove = carry['arrayLeafAbove']
                arrayLeafBelow = carry['arrayLeafBelow']
                arrayGapter = carry['arrayGapter']
                indexLeafActive = carry['indexLeafActive']

                arrayLeafAbove = arrayLeafAbove.at[indexLeafActive].set(arrayGap[indexGapActive])
                arrayLeafBelow = arrayLeafBelow.at[indexLeafActive].set(arrayLeafBelow[arrayLeafAbove[indexLeafActive]])
                arrayLeafBelow = arrayLeafBelow.at[arrayLeafAbove[indexLeafActive]].set(indexLeafActive)
                arrayLeafAbove = arrayLeafAbove.at[arrayLeafBelow[indexLeafActive]].set(indexLeafActive)
                arrayGapter = arrayGapter.at[indexLeafActive].set(indexGapActive)
                indexLeafActive += 1

                carry = {**carry,
                         'arrayLeafAbove': arrayLeafAbove,
                         'arrayLeafBelow': arrayLeafBelow,
                         'arrayGapter': arrayGapter,
                         'indexLeafActive': indexLeafActive,
                         'indexGapActive': indexGapActive}
                return carry 

            carry = jax.lax.cond(indexLeafActive > 0, true_branch_leaf, lambda c: c, carry)
            return carry 

        final_carry = jax.lax.while_loop(activeLeafGT0, countFoldings, carry)
        return final_carry['totalFoldings']

    hubris_partial = functools.partial(
        hubris,
        computationDivisions=computationDivisions,
        countLeavesTotal=countLeavesTotal,
        countDimensionsTotal=countDimensionsTotal,
        arrayD=arrayD
    )
    totalFoldingsArray = jax.vmap(hubris_partial)(arrayIndicesComputation)
    foldingsTotal = int(totalFoldingsArray.sum())
    return foldingsTotal