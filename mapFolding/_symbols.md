# Reading and typing the symbols

You only need to recognize a small set of symbols to read `mapFolding`. The names below give
you handles for discussing the glyphs; you can use the notation without learning another language.
For the coordinate model and the head/tail convention, read
[Semiotics in mapFolding](_semiotics.md).

(AI generated content)

## The numeral glyphs name indices from zero through nine

| Glyph | Name | Meaning |
| ----- | ---- | ------- |
| `零`  | ling | Zero    |
| `一`  | yi   | One     |
| `二`  | er   | Two     |
| `三`  | san  | Three   |
| `四`  | si   | Four    |
| `五`  | wu   | Five    |
| `六`  | liu  | Six     |
| `七`  | qi   | Seven   |
| `八`  | ba   | Eight   |
| `九`  | jiu  | Nine    |

Remember `一`, `二`, and `三` by their one, two, and three horizontal strokes. The two
strokes of `八` open apart. For any numeral you do not recognize, the snippets let you
search with its familiar digit: `Z number 6` inserts `六`.

These are index names in the positional notation. For example, imported `二` selects the
dimension at tail index two. Its coordinate has `1` in that position. The digit `2` is not a
coordinate value. [The coordinate guide](_semiotics.md) explains this distinction in detail.

## The surrounding glyphs tell you how to read an identifier

| Glyph  | Name   | Role in this package                              | Example                          |
| ------ | ------ | ------------------------------------------------- | -------------------------------- |
| `首`   | shou   | Head; count or inspect from the head.             | `首零`, `工dimension首一`        |
| `次`   | ci     | An index or place in an ordering.                 | `次Dimension`, `次diagonal`      |
| `形`   | xing   | A type or form.                                   | `形ArrayArcCode1D`, `形Triangle` |
| `吗`   | ma     | A question, usually answered by a Boolean.        | `isOdd吗`, `leaf吗`              |
| `工`   | gong   | A callable that produces the named value.         | `工dimensionTail`                |
| `名`   | ming   | A name, such as an identifier stored as text.     | `名Module`                       |
| `个`   | ge     | An unspecified item or generic type.              | `getLeaf[个]`                    |
| `是`   | shi    | “Is” or “to be”; read the rest of the expression. | `是valid`                        |
| `上`   | shang  | A superscript or exponent.                        | `mapShapeIs2上nDimensions`       |
| `下`   | xia    | A subscript or indexed specialization.            | `n下k`                           |
| `小于` | xiaoyu | Less than, `<`; `小` means small.                 | `k__小于__r`                     |

These roles are package conventions. Recognizing `吗` as a question or `形` as a type often
lets you read the English part of an identifier without pausing over the glyph.

For `吗`, use the author's visual mnemonic: it looks like a question mark. For `上` and `下`,
notice how the small stroke sits above or below the long horizontal stroke. These are memory
aids, not explanations of how the characters developed.

The snippets also offer the matching comparison `大于` (dayu), greater than, `>`. Remember
`大` (da, big) as a person holding their arms out and saying, “It was this big!” The pair
`小于` and `大于` shares `于`; changing small to big changes the comparison.

## Every snippet starts with `Z` and can be found with English words

The repository supplies [`identifiers.code-snippets`][snippets].
VS Code loads the file from the repository's `.vscode` directory. You can also copy it into
another project's `.vscode` directory, or use **Snippets: Configure Snippets** to create a
global snippets file and copy its contents there. VS Code's
[snippet documentation](https://code.visualstudio.com/docs/editing/userdefinedsnippets)
describes both scopes.

Every prefix starts with the letter `Z`. Start with `Z`, request suggestions with
**Ctrl+Space**, and use English words to narrow the choices. You do not need to remember a
glyph's name or install a Chinese keyboard.

| Search prefix        | Inserted glyph |
| -------------------- | -------------- |
| `Z number 2`         | `二`           |
| `Z order First`      | `首`           |
| `Z order Index`      | `次`           |
| `Z Question`         | `吗`           |
| `Z types Type`       | `形`           |
| `Z name`             | `名`           |
| `Z types Expression` | `工`           |

Some snippet descriptions also describe uses in `astToolkit`. The `Expression` prefix finds
`工`, whose role in `mapFolding` identifiers is to produce the named value. The snippet file
includes additional AST vocabulary; you do not need that vocabulary to understand map
coordinates.

Outside VS Code, copy glyphs from this reference or from existing code. Each symbol is part
of the Python identifier, so preserve it when copying a name or importing an object.

## A font with coordinated character widths keeps mixed identifiers aligned

A font can display Latin letters well and still lack the CJK glyphs used here. An editor
then substitutes another font for those glyphs, which can disturb spacing and alignment.
Choose a programming font with its own CJK coverage and coordinated character widths.

These families provide options for a uniform layout.

| Font                                  | Variant to look for                                              |
| ------------------------------------- | ---------------------------------------------------------------- |
| [Integrated Code 火][integrated-font] | A CJK variant, such as `Integrated Code 火 简化字`.              |
| [Sarasa Gothic][sarasa-font]          | A programming variant, such as `Sarasa Mono SC`.                 |
| [Maple Mono][maple-font]              | `Maple Mono CN` or `Maple Mono NF CN` for included CJK coverage. |

Integrated Code 火 is Hunter's font; download the fonts from its Releases page. Sarasa also
offers Term and Fixed programming variants.

Coordinated widths need not make every glyph equally wide. Maple Mono's CN variants, for
example, align one CJK glyph with two Latin character cells. A consistent width relationship
keeps mixed text on a predictable grid.

Install your chosen font and select its installed family name in VS Code's **Editor: Font
Family** setting. The repository's
[`mapFolding.code-workspace`][workspace]
contains a font preference list and enables ligatures. Opening that workspace applies its
settings; listing a font does not install it. Choose the same family separately in your
terminal if you also want the terminal's mixed text to align.

## Both guides are available beside the installed Python modules

This reference is `_symbols.md`; the conceptual guide is [_semiotics.md](_semiotics.md).
Both are included in the Python distribution and installed inside `mapFolding`. After
`import mapFolding`, `print(mapFolding.__file__)` locates their directory.

[snippets]:
  https://github.com/hunterhogan/mapFolding/blob/main/.vscode/identifiers.code-snippets
[workspace]:
  https://github.com/hunterhogan/mapFolding/blob/main/.vscode/mapFolding.code-workspace
[integrated-font]: https://github.com/hunterhogan/Integrated_code_fire_font
[sarasa-font]: https://github.com/be5invis/Sarasa-Gothic
[maple-font]: https://github.com/subframe7536/maple-font
