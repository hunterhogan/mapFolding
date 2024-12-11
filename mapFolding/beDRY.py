from typing import List
import math

def getLeavesTotal(listDimensions: List[int]) -> int:
    """
    Calculate the product of non-zero, non-negative integers in the given list.

    Parameters:
        listDimensions: A list of integers representing dimensions.

    Returns:
        productDimensions: The product of all positive integer dimensions in the list.
        Returns 0 if all dimensions are 0.

    Raises:
        ValueError: If the list contains negative integers or non-integer values.
        TypeError: If input is not a list or contains invalid types.
    """
    if not isinstance(listDimensions, list):
        raise TypeError(f"listDimensions must be a list, not {type(listDimensions)}")
    if not listDimensions:
        raise ValueError("listDimensions must not be empty")
    
    try:
        if any(not isinstance(dimension, int) or dimension < 0 for dimension in listDimensions):
            raise ValueError(f"listDimensions must contain only non-negative integers")
    except TypeError:
        raise ValueError(f"listDimensions contains invalid types")

    listDimensionsNonZero = [dimension for dimension in listDimensions if dimension > 0]
    if not listDimensionsNonZero:
        return 0
    return math.prod(listDimensionsNonZero)

