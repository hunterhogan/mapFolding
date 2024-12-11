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
        try:
            # Let Python raise TypeError for non-iterables
            iterator = iter(listDimensions)  # Will fail fast for non-iterables
            
            # Store initial length to detect modifications
            lengthInitial = len(listDimensions)
            
            for dimension in listDimensions:
                if len(listDimensions) != lengthInitial:
                    raise RuntimeError("Input sequence was modified during iteration")
                    
                # Explicitly check for integer type, but allow conversion from float if it's a whole number
                if isinstance(dimension, bool):  # Check for boolean first since it's a subclass of int
                    raise TypeError(f"Boolean values ({dimension}) are not allowed as dimensions")
                
                if isinstance(dimension, (int, float)):
                    if float(dimension).is_integer():
                        dimension = int(dimension)  # Convert float to int if whole number
                    else:
                        raise ValueError(f"Dimension {dimension} must be a whole number")
                else:
                    raise TypeError(f"Dimension {dimension} must be numeric")
                    
                if dimension < 0:
                    raise ValueError(f"Element {dimension} must be non-negative")

            # Convert to list to handle all sequence types uniformly
            listDimensionsNonZero = [int(d) for d in listDimensions if d > 0]
            if not listDimensionsNonZero:
                return 0
                
            productDimensions = 1
            for dimension in listDimensionsNonZero:
                if dimension > sys.maxsize // productDimensions:
                    raise OverflowError("Product would exceed maximum integer size")
                productDimensions *= dimension
                
            return productDimensions
            
        except TypeError as ERRORtype:
            if not hasattr(listDimensions, '__iter__'):
                ERRORmessage = f"Input must be iterable, not {type(listDimensions)}"
            else:
                ERRORmessage = f"Invalid element in listDimensions: {ERRORtype.args[0]}"
            raise TypeError(ERRORmessage) from None
