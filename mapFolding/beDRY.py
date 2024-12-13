from typing import List
from Z0Z_tools import intInnit
import sys

def parseListDimensions(listDimensions: List[int], parameterName: str = 'unnamed parameter') -> List[int]:
    """
    Parse and validate a list of dimensions.
    This function takes a list of integers representing dimensions and validates that all dimensions
    are non-negative values. It first converts all elements to integers if possible, then checks
    if each dimension is non-negative.
    Parameters:
        listDimensions (List[int]): List of integers representing dimensions
        parameterName (str, optional): Name of the parameter for error messages. Defaults to 'unnamed parameter'
    Returns:
        List[int]: List of validated non-negative integers
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

