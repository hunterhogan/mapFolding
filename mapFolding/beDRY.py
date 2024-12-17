from typing import List, Tuple
from Z0Z_tools import intInnit
import sys

def parseListDimensions(listDimensions: List[int], parameterName: str = 'unnamed parameter') -> List[int]:
    """
    Parse and validate a list of dimensions.

    Parameters:
        listDimensions: List of integers representing dimensions
        parameterName ('unnamed parameter'): Name of the parameter for error messages. Defaults to 'unnamed parameter'
    Returns:
        listNonNegative: List of validated non-negative integers
    Raises:
        ValueError: If any dimension is negative or if the list is empty
        TypeError: If any element cannot be converted to integer (raised by parseListInt)
    """
    listValidated = intInnit(listDimensions, parameterName)
    listNonNegative = []
    for dimension in listValidated:
        if dimension < 0:
            raise ValueError(f"Dimension {dimension} must be non-negative")
        listNonNegative.append(dimension)

    if not listNonNegative:
        raise ValueError("At least one dimension must be non-negative")
    
    return listNonNegative

def getLeavesTotal(listDimensions: List[int]) -> int:
    """
    Calculate the product of non-zero, non-negative integers in the given list.

    Parameters:
        listDimensions: A list of integers representing dimensions.

    Returns:
        productDimensions: The product of all positive integer dimensions.
        Returns 0 if all dimensions are 0.
    """
    listNonNegative = parseListDimensions(listDimensions, 'listDimensions')
    listPositive = [dimension for dimension in listNonNegative if dimension > 0]
        
    if not listPositive:
        return 0
    else:
        productDimensions = 1
        for dimension in listPositive:
            if dimension > sys.maxsize // productDimensions:
                raise OverflowError("Product would exceed maximum integer size")
            productDimensions *= dimension
                
        return productDimensions

def validateListDimensions(listDimensions: List[int]) -> List[int]:
    """
    Validates and processes a list of dimensions.

    This function ensures that the input list of dimensions is not None,
    parses it to ensure all dimensions are non-negative, and then filters
    out any dimensions that are not greater than zero. If the resulting
    list has fewer than two dimensions, a NotImplementedError is raised.

    Parameters:
        listDimensions (List[int]): A list of integer dimensions to be validated.

    Returns:
        List[int]: A list of positive dimensions.

    Raises:
        ValueError: If the input listDimensions is None.
        NotImplementedError: If the resulting list of positive dimensions has fewer than two elements.
    """
    if listDimensions is None:
        raise ValueError(f"listDimensions is a required parameter.")
    listNonNegative = parseListDimensions(listDimensions, 'listDimensions')
    listPositive = [dimension for dimension in listNonNegative if dimension > 0]
    if len(listPositive) < 2:
        from typing import get_args
        from mapFolding.oeis import OEISsequenceID
        raise NotImplementedError(f"This function requires listDimensions, {listDimensions}, to have at least two dimensions greater than 0. Other functions in this package implement the sequences {get_args(OEISsequenceID)}. You may want to look at https://oeis.org/.")
    listDimensions = listPositive
    return listDimensions

def validateTaskDivisions(computationDivisions: int, computationIndex: int, n: int) -> Tuple[int, int]:
    """
    Validates the task divisions for a computation process.

    Parameters:
        computationDivisions (int): The number of divisions for the computation.
        computationIndex (int): The index of the current computation division.
        n (int): The total number of leaves.

    Returns:
        Tuple[int, int]: A tuple containing the validated computationDivisions and computationIndex.

    Raises:
        ValueError: If computationDivisions is greater than n.
        ValueError: If computationIndex is greater than or equal to computationDivisions when computationDivisions is greater than 1.
        ValueError: If computationDivisions or computationIndex are negative or not integers.
    """
    if computationDivisions > n:
        raise ValueError(f"computationDivisions, {computationDivisions}, must be less than or equal to the total number of leaves, {n}.")
    if computationDivisions > 1 and computationIndex >= computationDivisions:
        raise ValueError(f"computationIndex, {computationIndex}, must be less than computationDivisions, {computationDivisions}.")
    if computationDivisions < 0 or computationIndex < 0 or not isinstance(computationDivisions, int) or not isinstance(computationIndex, int):
        raise ValueError(f"computationDivisions, {computationDivisions}, and computationIndex, {computationIndex}, must be non-negative integers.")
    return computationDivisions, computationIndex

def validateParametersFoldings(listDimensions: List[int], computationDivisions: int, computationIndex: int) -> Tuple[List[int], int, int, int]:
    """
    Validates and processes the parameters for the folding computation.

    Parameters:
        listDimensions (List[int]): A list of dimensions for the folding task.
        computationDivisions (int): The number of divisions for the computation task.
        computationIndex (int): The index of the current computation task.

    Returns:
        listDimensions,computationDivisions,computationIndex,leavesTotal: A tuple containing the validated list of dimensions,
                                         the validated number of computation divisions,
                                         the validated computation index, and the total number of leaves.
    """
    listDimensions = validateListDimensions(listDimensions)
    leavesTotal = getLeavesTotal(listDimensions)
    computationDivisions, computationIndex = validateTaskDivisions(computationDivisions, computationIndex, leavesTotal)
    return listDimensions, computationDivisions, computationIndex, leavesTotal
