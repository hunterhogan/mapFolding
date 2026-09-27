# Integrating A400429

This inventory covers code that walks Dyck paths, advances noncrossing boundary states, or counts those states. The semi-meander triangle is [OEIS A400429](https://oeis.org/A400429); `triangleSemi` remains in some filenames and identifiers. The long-term goal is to integrate the useful research code into the main `mapFolding` package.

## Immediate goals

- Use kitAST and `astToolkit` to generate `pruneNumPy` and `prunePandas` flows. Call `shortcut` before the dictionary is converted. Create new code to implement shortcut functionality during counting. The new handwritten code can be in `matrixMeandersShare.py` or a separate module such as with C:\apps\mapFolding\mapFolding\algorithms\foldsSymmetric.py. In kitAST, you can either import the function or use ast to add the function to the module. You probably want to add it to the module with numpy so you can use the global variables there.
- Add the flows to `basecamp.countMeanders`.
- Add new parameters to the existing pytest tests. No new test functions.
- Update this document.

## Soon

- update pandas with latest optimization ideas.

## Ongoing

- If a module in mapFolding imports from C:\apps\mapFolding\research\matrixMeanders\infoBooth.py, move the information from infoBooth to a more appropriate place in mapFolding.

## Verification during transition

- [triangleSemiCheck.py](../../easyRun/triangleSemiCheck.py) is the verification module for modules in transition. Run it from the repository root after activating `.venv` with `python easyRun/triangleSemiCheck.py`.
- Use pytest for stable modules. Expand its coverage when transition modules become stable.

## Meander transfer engines

- [matrixMeanders.py](../../mapFolding/algorithms/matrixMeanders.py) is the hand-written baseline algorithm. [matrixMeandersNumPy.py](../../mapFolding/algorithms/matrixMeandersNumPy.py) is mostly hand-written and uses ndarray. [matrixMeandersPandas.py](../../mapFolding/algorithms/matrixMeandersPandas.py) is mostly handwritten and uses DataFrame.
- [matrixMeandersShare.py](../../mapFolding/algorithms/matrixMeandersShare.py) has handwritten functions not in the baseline implementation or specific to numpy or pandas. Its `shortcut` removes states whose A400429 diagonal contributions are supplied by known formulas.
- [dataBaskets.py](../../mapFolding/dataBaskets.py) defines the shared `StateMeanders` state.
- [basecamp.py](../../mapFolding/basecamp.py) is the package API.
- [makeModules.py](../../mapFolding/kitAST/matrixMeanders/makeModules.py) generates
  [bigInt.py](../../mapFolding/synthesized/matrixMeanders/bigInt.py) and
  [prune.py](../../mapFolding/synthesized/matrixMeanders/prune.py) from the Python transfer, and a
  [compiled Dyck helper](../../mapFolding/synthesized/matrixMeanders/matrixMeandersShare.py) from its
  Dyck walk. `makePrune` transforms the complete baseline module with `NodeChanger`, retaining
  its Dyck walk and imports, and inserts `state = shortcut(state)` before counting and after each
  boundary transfer. `IngredientsModule` keeps the complete transformed body and manages its
  import ledger and writing. Regeneration propagates baseline changes to both `bigInt` and `prune`.
  [theSSOT.py](../../mapFolding/kitAST/theSSOT.py) names the source and generated modules;
  [synthesizeModules.py](../../easyRun/synthesizeModules.py) invokes the generator.

## Semi-meander triangle: A400429

- `makeLookupDiagonal` probably needs a better home.
- [formulasTriangle/](formulasTriangle/) supplies the known diagonal formulas used for pruning and row construction. The transfer depends on these research formulas.
- [triangleSemiMake.py](../../easyRun/triangleSemiMake.py) makes selected A400429 rows or diagonals through values edited in the script. Its row mode uses formulas or individual-cell transfer; its diagonal mode passes the state from `countDiagonal` to `basecamp.countMeanders`. [infoBooth.py](infoBooth.py) and [factsBucketsSignatures.py](factsBucketsSignatures.py) supply output paths and measured state data.
- [countMeanders.py](../../easyRun/countMeanders.py) runs the package meander flows.
- All scripts select work through code edited in the IDE; they have no CLI selector.
