# Inefficient division of work

The difference between this version and the base version is the very clever statement

```python
if mod == 0 or l != mod or m % mod == res:
```

It is very difficult to divide the counting algorithm into independent tasks, but the modulo
operation is somewhat successful. Based on my observations and some very rough measurements,
calculating the total count by summing modulo division requires at least three times as much
total work (for maps with dimensions 2 x N) and (N-2)-times as much work (for N x N maps).

The maximum number of divisions is equal to the number of leaves on the map, but you can divide
the work into fewer divisions. (The modulo is very clever.)

To use it, call `foldings` with additional parameters: how many divisions you want, `mod` and which division
you want to calculate, `res`.

```python
foldingsTotal = foldings([2,9], 18, 4)
```
