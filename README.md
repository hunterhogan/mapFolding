# Python implementation of "Multi-dimensional map-folding"

`mapFolding.py` explicitly implements [The On-Line Encyclopedia of Integer Sequences](https://oeis.org/):

- [A001415](https://oeis.org/A001415) Number of ways of folding a 2 X n strip of stamps.
- [A001416](https://oeis.org/A001416) Number of ways of folding a 3 X n strip of stamps.
- [A001417](https://oeis.org/A001417) Number of ways of folding a 2 X 2 X ... X 2 n-dimensional map.
- [A001418](https://oeis.org/A001418) Number of ways of folding an n X n sheet of stamps.

`mapFolding.foldings()` will accept arbitrary values for the array dimensions, however, due to Python's zero-indexing, to count the folds of an *n X m* map, you almost certainly will not use an *n X m* array. Compare with `mapFolding.getDimensions()`.

## Code history

### Original code

You can find a transcription of the code in `foldings.AA`.

W. F. Lunnon, Multi-dimensional map-folding, *The Computer Journal*, Volume 14, Issue 1, 1971, Pages 75–80, [https://doi.org/10.1093/comjnl/14.1.75](https://doi.org/10.1093/comjnl/14.1.75) ([BibTex](citations/Lunnon.bibtex))

### Java implementation by archmageirvine

A Java implementation of [sequence A001415 of the On-Line Encyclopedia of Integer Sequences](https://oeis.org/A001415) by [Sean A. Irvine](https://github.com/archmageirvine/joeis/blob/80e3e844b11f149704acbab520bc3a3a25ac34ff/src/irvine/oeis/a001/A001415.java)
 includes the comments:

```java
/**
 * A001415 Number of ways of folding a 2 X n strip of stamps.
 * @author Fred Lunnon (ALGOL68, C versions)
 * @author Sean A. Irvine (Java port)
 */
...
  // Implements algorithm as described in "Multi-dimensional map-folding",
  // by W. F. Lunnon, The Computer J, 14, 1, pp. 75--80.  Note the original
  // paper contains a few omissions, so this actual code is based on a C
  // implementation by Fred Lunnon.
```

### Related Video

"How Many Ways Can You Fold a Map?" by Physics for the Birds, 2024 November 13
[![How Many Ways Can You Fold a Map?](https://img.youtube.com/vi/sfH9uIY3ln4/0.jpg)](https://www.youtube.com/watch?v=sfH9uIY3ln4)
