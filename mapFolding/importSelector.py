# useLovelace = True
useLovelace = False

if useLovelace:
    from mapFolding.lovelace import countSequential
    from mapFolding.lovelace import countParallel
    from mapFolding.lovelace import countInitialize

else:
    from mapFolding.countSequential import countSequential
    from mapFolding.countParallel import countParallel
    from mapFolding.countInitialize import countInitialize
