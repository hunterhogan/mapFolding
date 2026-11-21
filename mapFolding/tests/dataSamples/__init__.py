"""Samples shared by the mapFolding tests, and facts about them, named by what they hold.

- albums.py: Every folding of small strips of stamps and meander systems
- callables.py: Callables whose behavior a contract observes
- maps2上nDimensional.py: Facts about every folding of the 2^n-dimensional maps
- measure.py: The values of each function in `mapFolding.measure`
- oeis.py: Static values of the stored OEIS b-files, written by `mapFolding.kitAST.oeisModules.makeModulesOEIS`
- sourcesPython.py: Python source texts for the syntax transformations of `mapFolding.kitAST`
- triangles.py: Triangles of integers, and a sequence that forms a triangle
- arrayFoldings2上{n}Dimensional.pkl: The enumerated foldings, read by `mapFolding.kitFilesystem.getDataFrameFoldings`

Each test module holds its own parametrization. Consumers do not mutate any object here.
"""
