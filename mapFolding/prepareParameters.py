def countMinimumParsePoints(dimensions: list[int]) -> int:
    leavesTotal = 1
    for dimensionSize in dimensions:
        leavesTotal *= dimensionSize
    COUNTreachesParsePoint = sum(1 for potentialDivision in range(1, leavesTotal + 1) 
                                if any(potentialDivision == 1 or potentialDivision - dimensionSize >= 1 
                                      for dimensionSize in dimensions))
    return COUNTreachesParsePoint

def getDimensions(series: str, X_n: int) -> list[int]:  
    if isinstance(series, int):
        series = str(series)
    if series == '2':
        return [2, X_n]
    elif series == '3':
        return [3, X_n]
    elif series.lower() == '2 x 2':
        return [2] * X_n
    elif series == 'n':
        return [X_n + 1, X_n + 1]
    else:
        return [int(series), X_n]
