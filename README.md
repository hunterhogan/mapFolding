# Algorithm(s) for counting distinct ways to fold a map (or a strip of stamps)

The function `mapFolding.countFolds()` counts distinct ways to fold maps and strips of stamps. The function accepts two or more dimensions:

```python
from mapFolding import countFolds
foldsTotal = countFolds( [2,10] )
```

The directory [mapFolding/reference](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/reference) has

- a verbatim transcription of Lunnon's "procedure" published in 1971 by _The Computer Journal_,
- multiple referential versions of the procedure with explanatory comments including
- [hunterNumba.py](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/reference), a one-size-fits-all, self-contained, reasonably fast, contemporary algorithm that is nevertheless infected by _noobaceae ignorancium_, and
- miscellaneous notes.

[![pip install mapFolding](https://img.shields.io/badge/pip%20install-mapFolding-gray.svg?colorB=3b434b)](https://pypi.org/project/mapFolding/) [![Python Tests](https://github.com/hunterhogan/mapFolding/actions/workflows/pythonTests.yml/badge.svg)](https://github.com/hunterhogan/mapFolding/actions/workflows/pythonTests.yml) [![Static Badge](https://img.shields.io/badge/stinkin'%20badges-don't%20need-b98e5e)](https://youtu.be/g6f_miE91mk&t=4) ![PyPI - Downloads](https://img.shields.io/pypi/dd/mapFolding) ![Static Badge](https://img.shields.io/badge/issues-I%20have%20them-brightgreen) ![GitHub repo size](https://img.shields.io/github/repo-size/hunterhogan/mapFolding)

## Simple, easy usage based on OEIS IDs

`mapFolding` directly implements some IDs from _The On-Line Encyclopedia of Integer Sequences_ ([BibTex](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/citations/oeis.bibtex) citation).

### Usage: command line

After installing (see below), `OEIS_for_n` will run a computation from the command line.

```cmd
(mapFolding) C:\apps\mapFolding> OEIS_for_n A001418 5
186086600 distinct folding patterns.
Time elapsed: 1.605 seconds
```

Use `getOEISids` to get the most up-to-date list of available OEIS IDs.

```cmd
(mapFolding) C:\apps\mapFolding> getOEISids

Available OEIS sequences:
  A001415: Number of ways of folding a 2 X n strip of stamps.
  A001416: Number of ways of folding a 3 X n strip of stamps.
  A001417: Number of ways of folding a 2 X 2 X ... X 2 n-dimensional map.
  A001418: Number of ways of folding an n X n sheet of stamps.
  A195646: Number of ways of folding a 3 X 3 X ... X 3 n-dimensional map.

Usage examples:
  Command line:
    OEIS_for_n A001415 8
  Python:
    from mapFolding import oeisIDfor_n
    foldsTotal = oeisIDfor_n('A001415', 8)
```

### Usage: Python module or REPL

Use `mapFolding.oeisIDfor_n()` to compute a(n) for an OEIS ID.

```python
from mapFolding import oeisIDfor_n
foldsTotal = oeisIDfor_n( 'A001418', 4 )
```

### An "advanced" and likely unnecessary feature: clear `mapFolding`'s cache of OEIS data

Clear _The On-Line Encyclopedia of Integer Sequences_ data from the `mapFolding` cache:

```sh
(mapFolding) C:\apps\mapFolding> clearOEIScache
Cache cleared from C:\apps\mapFolding\mapFolding\.cache
```

## Connections to "Multi-dimensional map-folding" by W. F. Lunnon

### The typo-laden algorithm published in 1971

The full paper, W. F. Lunnon, Multi-dimensional map-folding, _The Computer Journal_, Volume 14, Issue 1, 1971, Pages 75–80, [https://doi.org/10.1093/comjnl/14.1.75](https://doi.org/10.1093/comjnl/14.1.75) ([BibTex](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/citations/Lunnon.bibtex) citation) is available at the DOI link. (As of 3 January 2025, the paper is a PDF of images, not text, and can be accessed without cost or login.)

In [`foldings.txt`](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/reference/foldings.txt), you can find a text transcription of the algorithm as it was printed in 1971. In [`foldings.AA`](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/reference/foldings.AA), I have corrected obvious transcription errors, documented with comments, and I have reformatted line breaks and indentation. For contemporary readers, the result is likely easier to read than the text transcription or the original paper are easy to read. This is especially true if you view the document with semantic highlighting, such as with [Algol 60 syntax highlighter](https://github.com/PolariTOON/language-algol60).

### Java implementation(s) and improvements

[archmageirvine](https://github.com/archmageirvine/joeis/blob/80e3e844b11f149704acbab520bc3a3a25ac34ff/src/irvine/oeis/a001/A001415.java) ([BibTex](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/citations/jOEIS.bibtex) citation) says about the Java code:

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

## Map-folding Video

~~This caused my neurosis:~~ I enjoyed the following video, which is what introduced me to map folding.

"How Many Ways Can You Fold a Map?" by Physics for the Birds, 2024 November 13 ([BibTex](https://github.com/hunterhogan/mapFolding/blob/main/mapFolding/citations/Physics_for_the_Birds.bibtex) citation)

[![How Many Ways Can You Fold a Map?](https://i.ytimg.com/vi/sfH9uIY3ln4/hq720.jpg)](https://www.youtube.com/watch?v=sfH9uIY3ln4)

## Installation

```sh
pip install mapFolding
```

## My recovery

[![Static Badge](https://img.shields.io/badge/2011_August-Homeless_since-blue?style=flat)](https://HunterThinks.com/support)
[![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UC3Gx7kz61009NbhpRtPP7tw)](https://www.youtube.com/@HunterHogan)
