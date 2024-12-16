import jax
import jax.numpy as jnp
import functools

def spoon(computationDivisions: int, arrayIndicesComputation: jax.Array, countLeavesTotal: int, countDimensionsTotal: int, arrayD: jax.Array) -> jax.Array:
    """Computes total 'foldings' using JAX and lax.while_loop.

    Parameters:
        computationDivisions: the number of logical parts to which the full problem is divided.
        arrayIndicesComputation: array of indices: one for each logical part in `range(computationDivisions)`.
        countLeavesTotal: total leaves of the map.
        countDimensionsTotal: total dimensions of the map.
        arrayD: array with shape `(countDimensionsTotal + 1, countLeavesTotal + 1, countLeavesTotal + 1)`. Represents how leaf `indexLeaf` connects to leaf `indexM` in dimension `i` when inserting `indexLeaf`: `arrayD[i, indexLeaf, indexM]`.

    Returns:
        Array of total foldings.
    """

    def hubris(computationIndex, computationDivisions, countLeavesTotal, countDimensionsTotal, arrayD): 
        arrayLeafAbove = jnp.zeros(countLeavesTotal + 1, dtype=jnp.int64)
        arrayLeafBelow = jnp.zeros(countLeavesTotal + 1, dtype=jnp.int64)
        arrayCount = jnp.zeros(countLeavesTotal + 1, dtype=jnp.int64)
        arrayGapter = jnp.zeros(countLeavesTotal + 1, dtype=jnp.int64)
        arrayGap = jnp.zeros((countLeavesTotal * countLeavesTotal) + 1, dtype=jnp.int64)
        totalFoldings = 0
        indexGapActive = 0
        indexLeafActive = 1
        dimensionCountUnconstrained = 0 

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

        def condition(carry):
            return carry['indexLeafActive'] > 0

        def body(carry):
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

            condition_main = (indexLeafActive <= 1) | (arrayLeafBelow[0] == 1)

            def true_branch(carry):
                indexLeafActive = carry['indexLeafActive']
                totalFoldings = carry['totalFoldings']

                condition_overflow = indexLeafActive > countLeavesTotal

                def increment_foldings(carry):
                    totalFoldings = carry['totalFoldings'] + countLeavesTotal
                    carry = {**carry, 'totalFoldings': totalFoldings, 'dimensionCountUnconstrained': 0}
                    return carry

                def process_leaves(carry):
                    dimensionCountUnconstrained = 0
                    indexGapper = arrayGapter[indexLeafActive - 1]
                    indexGapActive = indexGapper
                    arrayGap = carry['arrayGap']
                    arrayCount = carry['arrayCount']

                    def fori_body(indexDimension, val):
                        dimensionCountUnconstrained, indexGapActive, indexGapper, arrayGap, arrayCount = val
                        conditionDimension = arrayD[indexDimension, indexLeafActive, indexLeafActive] == indexLeafActive
                        dimensionCountUnconstrained += jnp.where(conditionDimension, 1, 0)

                        indexM = arrayD[indexDimension, indexLeafActive, indexLeafActive]

                        def while_cond(state):
                            indexM, indexGapper, arrayGap, arrayCount = state
                            return indexM != indexLeafActive

                        def while_body(state):
                            indexM, indexGapper, arrayGap, arrayCount = state
                            conditionIndexM = (
                                (computationDivisions == 0) |
                                (indexLeafActive != computationDivisions) |
                                ((indexM % computationDivisions) == computationIndex)
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
                            indexM = arrayD[indexDimension, indexLeafActive, arrayLeafBelow[indexM]]
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
                        1, countDimensionsTotal + 1, fori_body, valInit
                    )
                    dimensionCountUnconstrained, indexGapActive, indexGapper, arrayGap, arrayCount = valFinal

                    condition_all_unconstrained = dimensionCountUnconstrained == countDimensionsTotal

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
                        condition_count = arrayCount[arrayGap[j]] == countDimensionsTotal - dimensionCountUnconstrained
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

            carry = jax.lax.cond(condition_main, true_branch, lambda c: c, carry)  #Note: lambda c: c does NOT modify carry

            def while_cond_inner(state):
                indexLeafActive, indexGapActive, arrayGapter, carry = state
                return (indexLeafActive > 0) & (indexGapActive == arrayGapter[indexLeafActive - 1])

            def while_body_inner(state):
                indexLeafActive, indexGapActive, arrayGapter, carry = state
                indexLeafActive -= 1
                arrayLeafBelow = carry['arrayLeafBelow']
                arrayLeafAbove = carry['arrayLeafAbove']
                arrayLeafBelow = arrayLeafBelow.at[arrayLeafAbove[indexLeafActive]].set(arrayLeafBelow[indexLeafActive])
                arrayLeafAbove = arrayLeafAbove.at[arrayLeafBelow[indexLeafActive]].set(arrayLeafAbove[indexLeafActive])
                carry = {**carry,
                        'arrayLeafAbove': arrayLeafAbove,
                        'arrayLeafBelow': arrayLeafBelow}
                return (indexLeafActive, indexGapActive, arrayGapter, carry)

            state_init = (indexLeafActive, indexGapActive, arrayGapter, carry)
            state_final = jax.lax.while_loop(while_cond_inner, while_body_inner, state_init)
            indexLeafActive, indexGapActive, arrayGapter, carry = state_final

            carry = {**carry, 'indexLeafActive': indexLeafActive, 'indexGapActive': indexGapActive}

            def true_branch_leaf(carry):
                indexGapActive = carry['indexGapActive'] - 1
                arrayGap = carry['arrayGap']
                arrayLeafAbove = carry['arrayLeafAbove']
                arrayLeafBelow = carry['arrayLeafBelow']
                indexLeafActive = carry['indexLeafActive']
                arrayLeafAbove = arrayLeafAbove.at[indexLeafActive].set(arrayGap[indexGapActive])
                arrayLeafBelow = arrayLeafBelow.at[indexLeafActive].set(arrayLeafBelow[arrayLeafAbove[indexLeafActive]])
                arrayLeafBelow = arrayLeafBelow.at[arrayLeafAbove[indexLeafActive]].set(indexLeafActive)
                arrayLeafAbove = arrayLeafAbove.at[arrayLeafBelow[indexLeafActive]].set(indexLeafActive)
                arrayGapter = carry['arrayGapter']
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
            carry = {**carry, 'totalFoldings': totalFoldings} #delete?
            return carry 

        final_carry = jax.lax.while_loop(condition, body, carry)
        return final_carry['totalFoldings']

    hubris_partial = functools.partial(hubris, computationDivisions=computationDivisions, countLeavesTotal=countLeavesTotal, countDimensionsTotal=countDimensionsTotal, arrayD=arrayD)
    return jax.vmap(hubris_partial)(arrayIndicesComputation)