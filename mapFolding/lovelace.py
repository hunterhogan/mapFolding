from numba import njit
import numpy

# I composed this module to be used with this toggle switch: https://marketplace.visualstudio.com/items?itemName=eliostruyf.vscode-hide-comments
from .lovelaceIndices import taskDivisions, taskIndex, leavesTotal, dimensionsTotal # Indices of array `the`. Static integer values
from .lovelaceIndices import A, B, count, gapter # Indices of array `track`. Dynamic values; each with length `leavesTotal + 1`
from .lovelaceIndices import incompleteTotal, activeGap1ndex, activeLeaf1ndex, unconstrainedLeaf, eniggma, amigoLeaf1ndex # Indices of array `an`. Dynamic integer values; `incompleteTotal` could be larger than 10^15 or 2^46, but the other numbers are less than leavesTotal + 1
from .lovelaceIndices import g, l, dd, gg, m # Alternative indices of array `an` to be more similar to the original code

@njit(cache=False) #cache=False cache=False cache=False cache=False cache=False cache=False cache=False cache=False cache=False cache=False 
def countFoldings(track: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], gap: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], the: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], D: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]]):
    an = _ = numpy.zeros(6, dtype=numpy.int64)
    _[l] = 1 # an[activeLeaf1ndex] = 1

    while _[l] > 0: # while an[activeLeaf1ndex] > 0:
        if _[l] <= 1 or track[B][0] == 1: 
            if _[l] > the[leavesTotal]:
                an[incompleteTotal] += the[leavesTotal]
            else:
                _[dd] = 0     
                # Track possible gaps 
                _[gg] = track[gapter][_[l] - 1]
                # Reset gap index
                _[g] = _[gg]

                # Count possible gaps for leaf _[l] in each section
                for i in range(1, the[dimensionsTotal] + 1):
                    if D[i][_[l]][_[l]] == _[l]:
                        _[dd] += 1
                    else:
                        _[m] = D[i][_[l]][_[l]]
                        while _[m] != _[l]:
                            if the[taskDivisions] == 0 or _[l] != the[taskDivisions] or _[m] % the[taskDivisions] == the[taskIndex]:
                                gap[_[gg]] = _[m]
                                if track[count][_[m]] == 0:
                                    _[gg] += 1
                                track[count][_[m]] += 1
                            _[m] = D[i][_[l]][track[B][_[m]]]

                # If leaf _[l] is unconstrained in all sections, it can be inserted anywhere
                if _[dd] == the[dimensionsTotal]: 
                    for _[m] in range(_[l]):
                        gap[_[gg]] = _[m]
                        _[gg] += 1

                # Filter gaps that are common to all sections
                for j in range(_[g], _[gg]): 
                    gap[_[g]] = gap[j]
                    if track[count][gap[j]] == the[dimensionsTotal] - _[dd]:
                        _[g] += 1
                    # Reset track[count] for next iteration
                    track[count][gap[j]] = 0  

        # Recursive backtracking steps
        while _[l] > 0 and _[g] == track[gapter][_[l] - 1]: 
            _[l] -= 1
            track[B][track[A][_[l]]] = track[B][_[l]]
            track[A][track[B][_[l]]] = track[A][_[l]]

        if _[l] > 0:
            _[g] -= 1
            track[A][_[l]] = gap[_[g]]
            track[B][_[l]] = track[B][track[A][_[l]]]
            track[B][track[A][_[l]]] = _[l]
            track[A][track[B][_[l]]] = _[l]
            # Save current gap index
            track[gapter][_[l]] = _[g]  
            # Move to next leaf
            _[l] += 1
    return an[incompleteTotal]
