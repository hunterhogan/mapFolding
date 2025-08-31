---
applyTo: '**/mmPandas.py'
---
Provide project context and coding guidelines that AI should follow when generating code, answering questions, or reviewing changes.

Use idiomatic pandas practices, ensuring code is efficient and leverages pandas' capabilities.

When modifying DataFrames:
- Avoid explicit loops; use vectorized operations or pandas built-in functions.
- Choose processes that use less memory, even if it requires more computation time.

Row counts:
- The initial dictionary will have between one and thirty items.
- The last DataFrame, after the groupby operation, will have exactly one row.
- On each iteration, the number of rows will increase until it hits a peak, then it will decrease.
- Within each iteration, the groupby operation will reduce the number of rows.
  - While the number of rows is increasing, the groupby will reduce the number of rows by a small amount.
  - When the number of rows is decreasing, the groupby will reduce the number of rows by a large amount.
- The peak number of rows is an estimated: rows = 0.71 * e**(0.4668 * bridges+1); using the value of bridges when passed as a parameter.
  - For bridges = 40, the peak is an estimated 146 million rows. MEMORY USAGE IS CRITICAL.
  - To discover a new distinctCrossings, the peak rows will be more than 1.5 billion.

No lambda functions.

https://pandas.pydata.org/pandas-docs/stable/reference/index.html
MCP context7 for documentation
