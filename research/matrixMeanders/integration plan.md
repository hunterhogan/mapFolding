# Integrating A400429

This inventory covers code that walks Dyck paths, advances noncrossing boundary states, or counts those states. The semi-meander triangle is [OEIS A400429](https://oeis.org/A400429); `triangleSemi` remains in some filenames and identifiers. The long-term goal is to integrate the useful research code into the main `mapFolding` package.

## Meander transfer engines

- [matrixMeanders.py](../../mapFolding/algorithms/matrixMeanders.py) is the hand-written dictionary transfer. It provides the Dyck bit walk and four boundary transitions.
- [matrixMeandersShare.py](../../mapFolding/algorithms/matrixMeandersShare.py) supplies closed, meander, and semi-meander starting states; compiled Dyck bit flipping; integer-width checks; and bucket sizing. Its `shortcut` removes states whose A400429 diagonal contributions are supplied by known formulas.
- [matrixMeandersNumPy.py](../../mapFolding/algorithms/matrixMeandersNumPy.py) uses arrays and memory-mapped storage, currently aggregating with `numpy.unique_inverse`. [matrixMeandersPandas.py](../../mapFolding/algorithms/matrixMeandersPandas.py) uses DataFrame grouping. Both can switch to the generated big-integer transfer when fixed-width values are too narrow.
- [basecamp.py](../../mapFolding/basecamp.py) selects the ordinary Python, NumPy, or Pandas flow. [dataBaskets.py](../../mapFolding/dataBaskets.py) defines their shared `StateMeanders` state.
- [makeModules.py](../../mapFolding/kitAST/matrixMeanders/makeModules.py) generates [bigInt.py](../../mapFolding/synthesized/matrixMeanders/bigInt.py) from the Python transfer and a [compiled Dyck helper](../../mapFolding/synthesized/matrixMeanders/matrixMeandersShare.py) from its Dyck walk. `makeNumPyChopItUp` remains unused. [theSSOT.py](../../mapFolding/kitAST/theSSOT.py) names the source and generated modules; [synthesizeModules.py](../../easyRun/synthesizeModules.py) invokes the generator.

## Semi-meander triangle: A400429

- [matrixMeandersTriangle.py](../../mapFolding/algorithms/matrixMeandersTriangle.py) has a separate hand-written transfer body with the four boundary transitions. It calls the shared `shortcut` after each layer and offers `countDiagonal` for individual A400429 cells.
- [formulasTriangle/](formulasTriangle/) supplies the known diagonal formulas used for pruning and row construction. The transfer depends on these research formulas.
- [triangleSemiCheck.py](../../easyRun/triangleSemiCheck.py) checks pruned A000682 semi-meander row totals and the triangle formulas. [triangleSemiMake.py](../../easyRun/triangleSemiMake.py) makes selected A400429 rows or diagonals through values edited in the script. Its row mode uses formulas or individual-cell transfer; its diagonal mode calls `countDiagonal`. [infoBooth.py](infoBooth.py) and [factsBucketsSignatures.py](factsBucketsSignatures.py) supply output paths and measured state data.
- [countMeanders.py](../../easyRun/countMeanders.py) runs the package meander flows. [reduceArches.py](../../easyRun/reduceArches.py) runs direct arch reduction and writes A287548 rows. These scripts select work through code edited in the IDE; they have no CLI selector.

## Other arch code

- [catalanArch.py](../../mapFolding/algorithms/catalanArch.py) enumerates complete Dyck arch configurations and counts their survival under reduction. It is a direct enumerator, not a transfer matrix.
- [archive/matrixMeanders/](../../archive/matrixMeanders/) retains predecessor dictionary, NumPy, file-backed, and Redis variants as references, not active entry points or generator outputs.

## Path into the main package

1. Choose a package-level interface for A400429 cells and rows and their OEIS identity. Keep the existing `triangleSemi` data filenames as compatibility names until their readers and writers can migrate together.
2. Move the triangle formula dependency out of `research/matrixMeanders` or expose it through a stable package module before treating the triangle transfer as part of the main package.
3. Deduplicate the hand-written transition bodies in the dictionary and triangle transfers, and then assess whether the NumPy and Pandas representations can use the same transition definition or generator. Keep their storage and aggregation choices explicit.
4. Decide which research scripts and measurements should become package entry points, examples, or tests. Preserve the distinction between A400429 triangle cells, A000682 row totals, and A287548 direct arch reduction.
