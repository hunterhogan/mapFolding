# JAX version

A JAX version would likely be faster than numba and it could automatically use GPU or TPU when available.

1. Building the data structures takes less than a millisecond, so I haven't optimized that step.
2. I convert the numpy.ndarray to jax.Array.
3. "pid.py" is supposed to be pure JAX but it didn't work.
4. I started over with "pider.py" as a hybrid of JAX and non-JAX.
   1. I make small changes and use the test modules to confirm the counts are correct.
   2. The hybrid module is painfully slow but the counts are correct.

While working on pider.py, I came up with a way to change improve parallelization, so I switched my focus to the Run Lola Run branch.
