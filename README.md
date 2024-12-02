# Python implementation of "Multi-dimensional map-folding"

`mapFolding` explicitly implements [The On-Line Encyclopedia of Integer Sequences](https://oeis.org/):

- [A001415](https://oeis.org/A001415) Number of ways of folding a 2 X n strip of stamps.
- [A001416](https://oeis.org/A001416) Number of ways of folding a 3 X n strip of stamps.
- [A001417](https://oeis.org/A001417) Number of ways of folding a 2 X 2 X ... X 2 n-dimensional map.
- [A195646](https://oeis.org/A195646) Number of ways of folding a 3 X 3 X ... X 3 n-dimensional map.
- [A001418](https://oeis.org/A001418) Number of ways of folding an n X n sheet of stamps.

`mapFolding.count_foldings()`, however, will accept arbitrary values for the array dimensions.

## Algorithm history

### The original algorithm

In [`foldings.txt`](mapFolding/reference/foldings.txt) and [`foldings.AA`](mapFolding/reference/foldings.AA), you can find transcriptions of the original algorithm as it was printed in 1971. The full paper is preserved as a PDF of images available at the DOI link below.

W. F. Lunnon, Multi-dimensional map-folding, *The Computer Journal*, Volume 14, Issue 1, 1971, Pages 75–80, [https://doi.org/10.1093/comjnl/14.1.75](https://doi.org/10.1093/comjnl/14.1.75) ([BibTex](mapFolding/citations/Lunnon.bibtex))

### ALGOL68, C, and Java versions

A Java implementation by [Sean A. Irvine](https://github.com/archmageirvine/joeis/blob/80e3e844b11f149704acbab520bc3a3a25ac34ff/src/irvine/oeis/a001/A001415.java) ([BibTex](mapFolding/citations/jOEIS.bibtex)) includes the comments:

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

## Related Video

"How Many Ways Can You Fold a Map?" by Physics for the Birds, 2024 November 13 ([BibTex](mapFolding/citations/Physics_for_the_Birds.bibtex))

[![How Many Ways Can You Fold a Map?](https://i.ytimg.com/vi/sfH9uIY3ln4/hq720.jpg)](https://www.youtube.com/watch?v=sfH9uIY3ln4)
