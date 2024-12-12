from typing import List
import sys

def getLeavesTotal(listDimensions: List[int]) -> int:
    """
    Calculate the product of non-zero, non-negative integers in the given list.

    Parameters:
        listDimensions: A list of integers representing dimensions.

    Returns:
        productDimensions: The product of all positive integer dimensions.
        Returns 0 if all dimensions are 0.

    Raises:
        Various built-in Python exceptions with enhanced error messages.
    """
    if not listDimensions:  # Empty list check is semantic, not type
        raise ValueError("listDimensions must not be empty")
    else:
        # Let Python raise TypeError for non-iterables in the following try/except block
        try:
            # Fail fast if input is not iterable - we don't use the iterator
            iter(listDimensions)  
            
            # Store initial length to detect modifications
            lengthInitial = len(listDimensions)
            
            for dimension in listDimensions:
                # Type checking - order matters due to bool being a subclass of int
                if isinstance(dimension, bool):
                    raise TypeError(f"Boolean values ({dimension}) are not allowed as dimensions")
                elif isinstance(dimension, (int, float)):
                    if float(dimension).is_integer():
                        dimension = int(dimension)  # Convert float to int if whole number
                    else:
                        raise ValueError(f"Dimension {dimension} must be a whole number")
                else:
                    raise TypeError(f"Dimension {dimension} must be numeric")
                    
                if dimension < 0:
                    raise ValueError(f"Element {dimension} must be non-negative")
                    
                # Check for modifications at the end of each iteration
                if len(listDimensions) != lengthInitial:
                    raise RuntimeError("Input sequence was modified during iteration")

            # Convert to list[int] to handle all sequence types uniformly
            listDimensionsNonZero = [int(d) for d in listDimensions if d > 0]
            
            if not listDimensionsNonZero:
                return 0
            else:
                productDimensions = 1
                for dimension in listDimensionsNonZero:
                    if dimension > sys.maxsize // productDimensions:
                        raise OverflowError("Product would exceed maximum integer size")
                    else:
                        productDimensions *= dimension
                    
                return productDimensions
            
        except TypeError as ERRORtype:
            if not hasattr(listDimensions, '__iter__'):
                ERRORmessage = f"{listDimensions=} does not have the '__iter__' attribute (it is not iterable), but it must have the '__iter__' attribute. {listDimensions} was passed as data type '{type(listDimensions)}'."
            else:
                ERRORmessage = f"Invalid element in listDimensions: {ERRORtype.args[0]}"
            raise TypeError(ERRORmessage) from None

