# Semiotics in mapFolding

The symbols in `mapFolding` let you read spatial relationships directly in the code.
This guide explains the coordinate notation from the beginning;
the [symbol reference](_symbols.md) explains how to recognize and type the symbols.

(AI generated content)

## These coordinates describe maps whose dimensions all have length two

A `2 × 2 × 2` map has three dimensions and eight leaves. A `2 × 2 × 2 × 2` map has four
dimensions and sixteen leaves. In general, `(2,) * n` describes an `n`-dimensional map with
`2^n` leaves. Each coordinate along each dimension is either `0` or `1`.

The positional symbols originated in the code dedicated to these maps. They still express
that specialized coordinate system. The rest of `mapFolding` can work with other map shapes;
these symbols do not provide a general purpose indexing system for those shapes.

## A leaf label represents a unique Cartesian coordinate in the unfolded map

For these maps, read a `Leaf` as a location in the unfolded map. Its sequence of coordinates
identifies one leaf, whether the map has three, four, seven, or many more dimensions. The
coordinate at the origin is `0` in every dimension; its label is `leafOrigin`.

Python stores the label as an `int`. The notation deliberately lets you set that fact aside.
A decimal label hides which dimensions have coordinate `1`. A positional numeral exposes
those coordinates, and the symbols name the dimension positions without requiring you
to translate the location into base ten.

For a four-dimensional map, the coordinate sequence `0101` describes one leaf: two dimensions
have coordinate `1`, and the other two have coordinate `0`. Keep all four positions in mind,
including the initial `0`. The map supplies the number of dimensions; the stored label alone
does not retain those initial zero coordinates.

## The tail and head replace familiar display directions with stable reference points

In an ordinary written binary numeral, the **tail** is at the right, where you may already
recognize the least significant bit, or LSB. The **head** is at the left, the most significant
end, often associated with the MSB. These familiar words are a bridge to the notation.
Thinking in terms of left and right afterward will cause problems: a display direction is
not an axis or a spatial relationship in a map with many dimensions. Use **tail** and **head**
as your reference points instead.

The tail anchors one end of the coordinate sequence. The head anchors the other end. Neither
the tail nor the head describes a physical orientation of the folded map. In the coordinate
constructors, the head belongs to the full sequence of `totalDimensions` coordinates, even
when the coordinate nearest the head is `0`.

## Both the tail and the head begin their positional indices at zero

The position nearest the tail is `零` (ling, zero); the next position from the tail is `一`
(yi, one); the next is `二` (er, two). Counting from the head starts again with `零`, then
`一`, then `二`. This symmetry is intentional. The index names the distance from the chosen
end, beginning at zero.

For four dimensions, each column below is one dimension position.

| Reference                   | Nearest the head | Next | Next | Nearest the tail |
| --------------------------- | ---------------- | ---- | ---- | ---------------- |
| Index counted from the tail | `三`             | `二` | `一` | `零`             |
| Index counted from the head | `零`             | `一` | `二` | `三`             |

The same position can therefore have two descriptions. In four dimensions, position `一`
from the tail is position `二` from the head. In `n` dimensions, tail index `k` corresponds
to head index `n - 1 - k`.

## A bare numeral symbol selects a dimension counted from the tail

An imported symbol such as `二` represents a coordinate sequence with `1` at that tail
position and `0` everywhere else. The symbol names the **index** of the dimension receiving
`1`; it does not state the stored decimal value or put the digit `2` into a coordinate.

| Symbol | Tail position receiving `1` | Four-dimensional coordinate sequence |
| ------ | --------------------------- | ------------------------------------ |
| `零`   | 0                           | `0001`                               |
| `一`   | 1                           | `0010`                               |
| `二`   | 2                           | `0100`                               |
| `三`   | 3                           | `1000`                               |

In particular, `零` selects tail position zero. It is not the origin. `leafOrigin` has `0`
in every position; `零` has `1` at the position nearest the tail.

The exported tail symbols run from `零` through `九`, covering tail indices zero through
nine. A map must have enough dimensions for the selected tail position to belong to it.
The coordinate idea itself is not limited to ten dimensions.

## The prefix `首` selects positions counted from the head

`首` means head. The function `首零(totalDimensions)` places `1` at the position nearest
the head. `首一(totalDimensions)` places `1` one position from the head. The head depends on
the full coordinate width, so these functions need `totalDimensions`.

| Expression    | Head positions receiving `1` | Four-dimensional coordinate sequence |
| ------------- | ---------------------------- | ------------------------------------ |
| `首零(4)`     | 0                            | `1000`                               |
| `首一(4)`     | 1                            | `0100`                               |
| `首零一(4)`   | 0 and 1                      | `1100`                               |
| `首零二(4)`   | 0 and 2                      | `1010`                               |
| `首一二三(4)` | 1, 2, and 3                  | `0111`                               |

Each numeral after `首` names a separate head position. Thus `首零二` selects head positions
zero and two; it does not mean position two written with an extra zero. The available
functions cover every nonempty combination of head positions zero through three. Use a
`totalDimensions` large enough to contain every named head position.

## Combining distinct positions builds a coordinate without a decimal translation

The package uses expressions such as `一 + 零` and `(零) + 首零(state.totalDimensions)`.
Read the first as coordinate `1` at tail positions one and zero. Read the second as coordinate
`1` at the position nearest the tail and at the position nearest the head.

For four dimensions, `一 + 零` gives `0011`, while `(零) + 首零(4)` gives `1001`.
The expressions expose the dimensions that matter to the algorithm.

Addition combines these coordinates as intended when the selected positions are distinct.
If a tail position and a head position refer to the same dimension, adding them produces
a carry instead of selecting that dimension twice. The width matters when combining tail
and head expressions.

## A leaf coordinate and a pile position retain different meanings

A `Leaf` identifies a panel of the unfolded map. A `Pile` identifies a position in the folded
stack. Moving a leaf to a different pile does not change the leaf's Cartesian coordinate.

Specialized algorithms also use the positional notation when expressing pile relationships.
Follow the surrounding `leaf`, `pile`, and type names to determine what an expression labels.
The shared numerical representation does not make a pile position a location in the unfolded
map. Similarly, `leafOrigin` and `pileOrigin` both have stored value zero but name different
roles.

## A coordinate position differs from the next position whose coordinate is one

The constructors `首零`, `首一`, and their combinations select fixed positions within the
map width. Every dimension position counts, including positions whose coordinate is `0`.

Some functions in `mapFolding.measure` instead locate dimensions whose coordinate is `1`.
For example, `工dimension首一` finds the second such dimension encountered from the head
of the supplied value and returns its tail-relative index. For `0101`, those occupied
positions have tail indices two and zero, so the second has tail index zero. This differs
from constructing `首一(4)`, which always selects head position one.

The function `dimensionIndex` recovers a tail-relative index from a single-dimension
coordinate such as `二`. It does not unpack a complete leaf coordinate. The individual
function docstrings describe these operations and their input requirements.

## The package exports the coordinate notation directly from `mapFolding`

The public imports are available directly from the package.

```python
from mapFolding import dimensionIndex, leafOrigin, pileOrigin, 一, 二, 零, 首零, 首零一
```

Earlier code imported the positional symbols from `mapFolding.semiotics`. Replace that import
with `from mapFolding import ...`. Their implementation now lives in `_semiotics.py`.

This guide and the [symbol reference](_symbols.md) ship inside the installed `mapFolding`
directory, beside `_semiotics.py`. To locate that directory in your Python environment, use
`print(mapFolding.__file__)` after `import mapFolding`. You can also read both guides through
the links after installation instructions in the main README.
