import sys
from typing import List

import numba
import numba.extending
from Z0Z_tools import intInnit


@numba.extending.overload(intInnit)
def intInnit_jitInnit(listDimensions, parameterName):
    if isinstance(listDimensions, numba.types.List) and isinstance(parameterName, numba.types.StringLiteral):
        def intInnit_jitInnit_implementInnit(listDimensions, parameterName):
            return listDimensions
        return intInnit_jitInnit_implementInnit
    return None

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

def validateListDimensions(listDimensions: List[int]) -> List[int]:
    """
    Validates and processes a list of dimensions.

    This function ensures that the input list of dimensions is not None,
    parses it to ensure all dimensions are non-negative, and then filters
    out any dimensions that are not greater than zero. If the resulting
    list has fewer than two dimensions, a NotImplementedError is raised.

    Parameters:
        listDimensions: A list of integer dimensions to be validated.

    Returns:
        listDimensionsPositive: A list of positive dimensions.

    Raises:
        ValueError: If the input listDimensions is None.
        NotImplementedError: If the resulting list of positive dimensions has fewer than two elements.
    """
    if not listDimensions:
        raise ValueError(f"listDimensions is a required parameter.")
    listNonNegative = parseListDimensions(listDimensions, 'listDimensions')
    listDimensionsPositive = [dimension for dimension in listNonNegative if dimension > 0]
    if len(listDimensionsPositive) < 2:
        raise NotImplementedError(f"This function requires listDimensions, {listDimensions}, to have at least two dimensions greater than 0. You may want to look at https://oeis.org/.")
    return listDimensionsPositive

def getLeavesTotal(listDimensions: List[int]) -> int:
    """
    Calculate the product of non-zero, non-negative integers in the given list.

    Parameters:
        listDimensions: A list of integers representing dimensions.

    Returns:
        productDimensions: The product of all positive integer dimensions. Returns 0 if all dimensions are 0.
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

numba.jit_module(cache=True)