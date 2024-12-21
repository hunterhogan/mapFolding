import numpy as NUMERICALPYTHON
from numpy.typing import NDArray
import jax
import jaxtyping

# Indices of array `the`. Static integer values
from mapFolding.piderIndices import taskDivisions, leavesTotal, dimensionsTotal 
# Indices of array `track`. Dynamic values; each with length `leavesTotal + 1`
from mapFolding.piderIndices import leafAbove, leafBelow, countDimensionsGapped, gapRangeStart

def spoon(connectionGraph: jax.Array, the: jax.Array, Z0Z_track: jax.Array, Z0Z_potentialGaps: jax.Array, arrayIndicesComputation: jax.Array):
    the = the.at[taskDivisions].set(0)
    taskIndex = arrayIndicesComputation[0]
    sherpaTrack = NUMERICALPYTHON.asarray(Z0Z_track).copy()
    sherpaPotentialGaps = NUMERICALPYTHON.asarray(Z0Z_potentialGaps).copy()

    def countFoldings(track: NDArray[NUMERICALPYTHON.int32], potentialGaps: NDArray[NUMERICALPYTHON.int32]):
        foldingsSubtotal = jax.numpy.int32(0)
        activeLeaf1ndex: int = 1 # index starts at 0, but 1ndex starts at 1
        activeGap1ndex: int = 0 # index starts at 0, but 1ndex starts at 1

        while activeLeaf1ndex > 0:
            if activeLeaf1ndex <= 1 or track[leafBelow][0] == 1:
                if activeLeaf1ndex > int(the[leavesTotal]):
                    foldingsSubtotal = jax.numpy.add(foldingsSubtotal, the[leavesTotal])
                else:
                    unconstrainedLeaf: int = 0
                    # Track possible gaps 
                    gap1ndexLowerBound = jax.numpy.int32(track[gapRangeStart][activeLeaf1ndex - 1])
                    # Reset gap index
                    activeGap1ndex = int(gap1ndexLowerBound)

                    # Count possible gaps for activeLeaf1ndex in each section
                    for dimension1ndex in range(1, int(the[dimensionsTotal]) + 1):
                        if connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex] == activeLeaf1ndex:
                            unconstrainedLeaf += 1
                        else:
                            leaf1ndexConnectee = connectionGraph[dimension1ndex][activeLeaf1ndex][activeLeaf1ndex]
                            while leaf1ndexConnectee != activeLeaf1ndex:
                                if int(the[taskDivisions]) == 0 or activeLeaf1ndex != int(the[taskDivisions]) or leaf1ndexConnectee % int(the[taskDivisions]) == int(taskIndex):
                                    potentialGaps[int(gap1ndexLowerBound)] = leaf1ndexConnectee
                                    addend = jax.numpy.where(jax.numpy.equal(track[countDimensionsGapped][leaf1ndexConnectee], 0), 1, 0)
                                    gap1ndexLowerBound = jax.numpy.add(gap1ndexLowerBound, addend)
                                    track[countDimensionsGapped][leaf1ndexConnectee] += 1
                                leaf1ndexConnectee = connectionGraph[dimension1ndex][activeLeaf1ndex][track[leafBelow][leaf1ndexConnectee]]

                    # If activeLeaf1ndex is unconstrained in all sections, it can be inserted anywhere
                    if unconstrainedLeaf == int(the[dimensionsTotal]):
                        for leaf1ndex in range(activeLeaf1ndex):
                            potentialGaps[int(gap1ndexLowerBound)] = leaf1ndex
                            gap1ndexLowerBound = jax.numpy.add(gap1ndexLowerBound, 1)

                    # Filter gaps that are common to all sections
                    for indexMiniGap in range(activeGap1ndex, int(gap1ndexLowerBound)):
                        potentialGaps[activeGap1ndex] = potentialGaps[indexMiniGap]
                        if track[countDimensionsGapped][potentialGaps[indexMiniGap]] == int(the[dimensionsTotal]) - unconstrainedLeaf:
                            activeGap1ndex += 1
                        # Reset track[count] for next iteration
                        track[countDimensionsGapped][potentialGaps[indexMiniGap]] = 0

            # Recursive backtracking
            while activeLeaf1ndex > 0 and activeGap1ndex == track[gapRangeStart][activeLeaf1ndex - 1]:
                activeLeaf1ndex -= 1
                track[leafBelow][track[leafAbove][activeLeaf1ndex]] = track[leafBelow][activeLeaf1ndex]
                track[leafAbove][track[leafBelow][activeLeaf1ndex]] = track[leafAbove][activeLeaf1ndex]

            if activeLeaf1ndex > 0:
                activeGap1ndex -= 1
                track[leafAbove][activeLeaf1ndex] = potentialGaps[activeGap1ndex]
                track[leafBelow][activeLeaf1ndex] = track[leafBelow][track[leafAbove][activeLeaf1ndex]]
                track[leafBelow][track[leafAbove][activeLeaf1ndex]] = activeLeaf1ndex
                track[leafAbove][track[leafBelow][activeLeaf1ndex]] = activeLeaf1ndex
                # Save current gap index
                track[gapRangeStart][activeLeaf1ndex] = activeGap1ndex 
                # Move to next leaf
                activeLeaf1ndex += 1
        return int(foldingsSubtotal)

    return int(countFoldings(sherpaTrack, sherpaPotentialGaps))