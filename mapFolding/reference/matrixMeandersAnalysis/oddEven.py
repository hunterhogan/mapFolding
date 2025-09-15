from math import exp as math_exp, log as math_log

def estimateBuckets(n: int, k: int) -> float:
    """Estimate the number of buckets for given integers n and k.

    Model form (log-space):
        log(estimate) = a
            + b * log(n)
            + c * log(k)
            + d * log(n)*log(k)
            + e * (k / n)
            + f * (n / k)

    Coefficients were obtained via ordinary least squares fit on log(buckets)
    using the dataset in df_oddEven.csv (38 observations). The regression achieved
    R^2 ≈ {coefficient_determination:.5f} (log space) with MAPE ≈ {mean_absolute_percentage_error:.2f}% on training data.

    NOTE: This is an empirical approximation; extrapolation outside the range
    n ∈ [{min(n_values)}, {max(n_values)}], k ∈ [{min(k_values)}, {max(k_values)}] may be unreliable.

    Parameters
    ----------
    n : int
        Primary size parameter (must be > 0)
    k : int
        Secondary size parameter (must be > 0)

    Returns
    -------
    float
        Estimated bucket count (positive real number). Caller may round if an
        integer is desired.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f'allegedInt n must be positive int, got {n!r}')
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f'allegedInt k must be positive int, got {k!r}')

    a = -679.088264366881
    b =  864.829109159972
    c = -873.871846814867
    d =    3.487829177620
    e =  943.512567960048
    f = -193.640628682536

    ln_n = math_log(n)
    ln_k = math_log(k)
    value_log = (a
                 + b * ln_n
                 + c * ln_k
                 + d * ln_n * ln_k
                 + e * (k / n)
                 + f * (n / k))
    return math_exp(value_log)
