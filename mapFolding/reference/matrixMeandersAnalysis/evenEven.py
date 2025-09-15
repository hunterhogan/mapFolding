"""
Matrix Meanders Buckets Estimation Formula.

This module provides a function to estimate the number of buckets in matrix meanders
analysis based on parameters n and k. The formula was derived through regression
analysis on empirical data and achieves an R² score of 0.9978.

Author: Generated from Jupyter notebook analysis
Date: September 2025
"""
import math

def estimateBucketsFromParametersNandK(parameterN: int, parameterK: int) -> float:
    """
    Estimate the number of buckets in matrix meanders analysis based on parameters n and k.

    This function implements a log-polynomial regression model derived from empirical data analysis
    of matrix meanders with even-even configurations. The model was trained on 34 data points
    and achieved an R² score of 0.9978, indicating excellent predictive performance.

    The underlying formula is:
    buckets = exp(0.087760 + 0.447340*n - 0.058715*k - 0.014116*n² + 0.072333*n*k - 0.090631*k²) - 1

    Parameters
    ----------
    parameterN : int
        The n parameter (must be positive integer, typically even)
    parameterK : int
        The k parameter (must be positive integer, typically even)

    Returns
    -------
    float
        Estimated number of buckets

    Raises
    ------
    ValueError
        If parameterN or parameterK are not positive integers

    Example:
    --------
    >>> estimateBucketsFromParametersNandK(20, 10)
    3656.4
    >>> estimateBucketsFromParametersNandK(44, 22)
    32715318.1

    Notes
    -----
    This formula is based on empirical data and is most accurate for values
    within the training range (n: 4-44, k: 2-22). Extrapolation beyond this
    range should be done with caution.

    The underlying mathematical relationship shows that buckets grow exponentially
    with both n and k, with complex polynomial interactions captured in the logarithmic space.

    Performance Metrics:
    - R² Score: 0.9978
    - Mean Absolute Error: 4.02% across all training examples
    - Median Error: 2.69%
    - Maximum Error: 18.86% (on smallest values)
    """
    # Input validation
    if not isinstance(parameterN, int) or parameterN <= 0:
        raise ValueError(f"parameterN must be a positive integer, got {parameterN}")
    if not isinstance(parameterK, int) or parameterK <= 0:
        raise ValueError(f"parameterK must be a positive integer, got {parameterK}")

    # Convert to float for calculations
    n = float(parameterN)
    k = float(parameterK)

    # Log-polynomial formula coefficients (derived from regression analysis)
    # Formula: log(buckets + 1) = intercept + c1*n + c2*k + c3*n² + c4*n*k + c5*k²
    interceptValue = 0.087760
    coefficientN = 0.447340
    coefficientK = -0.058715
    coefficientNSquared = -0.014116
    coefficientNTimesK = 0.072333
    coefficientKSquared = -0.090631

    # Calculate log(buckets + 1)
    logBucketsPlusOne = (interceptValue +
                        coefficientN * n +
                        coefficientK * k +
                        coefficientNSquared * (n * n) +
                        coefficientNTimesK * (n * k) +
                        coefficientKSquared * (k * k))

    # Transform back to original scale: buckets = exp(log(buckets + 1)) - 1
    estimatedBuckets = math.exp(logBucketsPlusOne) - 1.0

    return max(0.0, estimatedBuckets)  # Ensure non-negative result


def validateFormulaAccuracy() -> None:
    """
    Validate the formula against known test cases from the training data.

    This function demonstrates the accuracy of the formula by testing it
    against several known data points from the original dataset.
    """
    # Test cases from the original dataset (n, k, expected_buckets)
    testCases = [
        (4, 2, 4),
        (20, 10, 3592),
        (36, 18, 1666843),
        (44, 22, 35674291)
    ]

    print("Formula Validation Results:")
    print("-" * 50)
    print(f"{'n':<4} {'k':<4} {'Expected':<12} {'Predicted':<12} {'Error %':<8}")
    print("-" * 50)

    totalError = 0.0
    for n, k, expected in testCases:
        predicted = estimateBucketsFromParametersNandK(n, k)
        error = abs((expected - predicted) / expected) * 100
        totalError += error

        # Format large numbers appropriately
        expectedStr = f"{expected:.0f}" if expected < 1e6 else f"{expected:.2e}"
        predictedStr = f"{predicted:.1f}" if predicted < 1e6 else f"{predicted:.2e}"

        print(f"{n:<4} {k:<4} {expectedStr:<12} {predictedStr:<12} {error:<8.1f}")

    avgError = totalError / len(testCases)
    print("-" * 50)
    print(f"Average Error: {avgError:.2f}%")


if __name__ == "__main__":
    # Run validation when script is executed directly
    validateFormulaAccuracy()

    # Demonstrate usage
    print("\nUsage Examples:")
    print("-" * 30)

    examples = [(8, 4), (20, 10), (36, 18)]
    for n, k in examples:
        result = estimateBucketsFromParametersNandK(n, k)
        print(f"estimateBucketsFromParametersNandK({n}, {k}) = {result:.1f}")
