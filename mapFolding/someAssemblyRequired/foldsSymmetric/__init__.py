"""Generate synchronous and asynchronous symmetric-fold modules.

(AI generated docstring)

You can use this package to generate map-folding modules that count only symmetric folding
patterns. The package combines reusable syntax nodes with synchronous and asynchronous module
generators. The synchronous generator inserts symmetry filtering into generated algorithms, and
the asynchronous generator runs filtering concurrently with fold discovery.

Modules
-------
makeModulesFoldsSymmetric
    Generate synchronous symmetric-fold modules by transforming map-folding source modules.
makeModulesFoldsSymmetricAsynchronous
    Generate asynchronous symmetric-fold modules with concurrent symmetry filtering.
rawMaterialsFoldsSymmetric
    Provide reusable syntax nodes for generated symmetric-fold modules.
"""
