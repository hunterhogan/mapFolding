import pathlib
import os
from typing import Union
def countMinimumParsePoints(dimensions: list[int]) -> int:
    # TODO: improve this because the sizes of the divisions have a dramatically wide range.
    leavesTotal = 1
    for dimensionSize in dimensions:
        leavesTotal *= dimensionSize

    """
    If the number of `computationDivisions` is more than the number
    of times we reach the parse point, the concurrent processes will
    overcount the foldings. If a subtree will cross the parse point at
    least once, it is safe to divide the computation once. So, the minimum
    number of subtrees that can be parsed out is the maximum number of
    parsings: `computerDivisions`. *

    If this is true, perhaps generalize the calculation to k-degrees, then pass k as a parameter.

    Or, if the entire problem can be reliably divided into the exact number of divisions, then
    the first step should be to devolve the problem into the divisions. Second, use simplier 
    logic to count the foldings in a division. Third, sum the results of the divisions. Therefore,
    it's nearly certain that we don't know how to reliably calculate the number of divisions.

    But, is it possible to: calculate the 1st degree of divisions. Then each time the worker arrives at the 
    parse point, the worker stops, divides their current tree into one more degree of divisions, and
    "records" the entry point to those divisions.
    """
            
    COUNTreachesParsePoint = sum(1 for potentialDivision in range(1, leavesTotal + 1) 
                                if any(potentialDivision == 1 or potentialDivision - dimensionSize >= 1 
                                      for dimensionSize in dimensions))
    return COUNTreachesParsePoint

def getDimensions(series: str, X_n: int) -> list[int]:  
    """
    Return the dimensions of the array used to count the folds of a `series` `X_n` map.
    
    Explicitly implements dimensions for the following OEIS sequences
        A001415: 2 X n strip
        A001416: 3 X n strip
        A001417: 2 X 2 X ... X 2 (n-dimensional)
        A001418: n X n sheet
    """
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

def pathTasksToParameters(pathTasks: Union[str, os.PathLike[str]]) -> tuple[str, int, int, bool]:
    """
    Extracts the parameters from the pathTasks string.
    
    Parameters:
        pathTasks: A string containing the path to the task file.
    
    Returns:
        series,X_n,computationDivisions,normalFoldings: 
        The series type of the map, e.g. '2', '3', '2 X 2', 'n';
        The number of dimensions, n, for the specified series;
        The number of divisions to make in the computation;
        When True, enumerate only normal foldings.
    """
    series, X_n, computationDivisions, normalFoldings = pathlib.PurePosixPath(pathTasks).parts[-4:]
    return series, int(X_n), int(computationDivisions), bool(normalFoldings)

# * Or, I don't know what the hell I'm talking about.
