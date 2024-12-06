from typing import List

def countMinimumParsePoints(dimensions: List[int]) -> int:
    leavesTotal = 1
    for dimensionSize in dimensions:
        leavesTotal *= dimensionSize
            
    COUNTreachesParsePoint = sum(1 for potentialDivision in range(1, leavesTotal + 1) 
                                if any(potentialDivision == 1 or potentialDivision - dimensionSize >= 1 
                                      for dimensionSize in dimensions))
    return COUNTreachesParsePoint
