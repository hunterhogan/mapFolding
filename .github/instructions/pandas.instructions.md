---
applyTo: 'mapFolding/algorithms/*.py'
---
Use idiomatic pandas practices, ensuring code is efficient and leverages pandas's capabilities.

When modifying DataFrames:
- Avoid explicit loops; use vectorized operations or pandas built-in functions.
- Choose processes that use less memory, even if it requires more computation time.

No lambda functions.

Memory-efficient accumulation patterns:
- Avoid `if dataframe.empty:` checks - pandas operations naturally handle empty DataFrames with appropriate fill_value parameters.
- Avoid intermediate variables for groupby results - use direct assignment or chained operations.
- Avoid `.copy()` unless absolutely necessary for data integrity - direct assignment is preferred.
- Use `add(other, fill_value=0)` for accumulation instead of explicit empty checks and copies.
- Prefer in-place operations and direct DataFrame construction over creating temporaries.

https://pandas.pydata.org/pandas-docs/stable/reference/index.html
MCP context7 for documentation

"You should never modify something you are iterating over. This is not guaranteed to work in all cases." See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iterrows.html
