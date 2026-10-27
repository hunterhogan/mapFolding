"""Generate formula-source docstrings and selector annotations from metadata.

(AI generated docstring)

You can use this module to enrich sequence-formula source with descriptions and literal
selector values from `mapFolding.oeis.getMetadata` [1]. The generator writes source to a
filename derived by removing the configured prefix.

Contents
--------
Functions
	transformOEISidByFormula
		Write formula-source functions with metadata descriptions and selector annotations.
Variables
	moduleWarning
		Access the generated-file notice appended to the output module docstring.
	sourcePrefix
		Inspect the filename prefix removed when writing documented formula source.

References
----------
[1] `mapFolding.oeis.getMetadata`
"""
from __future__ import annotations

from astToolkit import Be, Grab, IfThis, Make, NodeChanger, NodeTourist, Then
from astToolkit.changeDef import makeDictionaryFunctionDef
from astToolkit.filesystem import parsePathFilename2astModule
from astToolkit.ohWow import unjoinBinOP
from hunterMakesPy import raiseIfNone
from hunterMakesPy.filesystemToolkit import writePython
from mapFolding.oeis import getMetadata
from typing import TYPE_CHECKING
import ast

if TYPE_CHECKING:
    from mapFolding.oeis._dataBaskets import MetadataOEISid
    from pathlib import Path

#------------------ General Settings ----------------------------------------------------------------------------------
sourcePrefix: str = 'zCuzDocStoopid'
"""Inspect the filename prefix removed when writing documented formula source.

(AI generated docstring)
"""

moduleWarning = "This is a generated file; edit the source file.\n"
"""Access the generated-file notice appended to the output module docstring.

(AI generated docstring)
"""

def transformOEISidByFormula(pathFilenameSource: Path) -> Path:
    """Write formula-source functions with metadata descriptions and selector annotations.

    (AI generated docstring)

    You can use this function to read formula source, obtain sequence metadata through
    `mapFolding.oeis.getMetadata` [1], insert function docstrings, and expand selector annotations
    with discovered literal values. The output filename removes `sourcePrefix` [2] from the stem.
    The function appends `moduleWarning` [3] to the module docstring and writes the resulting source.

    Sequence Function Selection
    ---------------------------
    The transformation selects seven-character names beginning with `A` and followed by six digits.
    The transformation expects a second positional parameter named `f` and collects match-value
    patterns as literal selectors. A new description is inserted before the existing body.

    Parameters
    ----------
    pathFilenameSource : Path
        Formula-source file containing documented sequence functions and a module docstring.

    Returns
    -------
    pathFilename : Path
        Path returned by the source writer after removing the configured filename prefix.

    Examples
    --------
    The module `easyRun.synthesizeModules` uses this call [4].

        ```python
        transformOEISidByFormula(pathFilenameSource)
        ```

    References
    ----------
    [1] `mapFolding.oeis.getMetadata`

    [2] `sourcePrefix`

    [3] `moduleWarning`

    [4] `easyRun.synthesizeModules`
    """
    pathFilenameWrite: Path = pathFilenameSource.with_stem(pathFilenameSource.stem.removeprefix(sourcePrefix))
    astModule: ast.Module = parsePathFilename2astModule(pathFilenameSource)
    dictionaryFunctionDef: dict[str, ast.FunctionDef] = makeDictionaryFunctionDef(astModule)

    for oeisID, FunctionDef in dictionaryFunctionDef.items():
        if oeisID.startswith('A') and len(oeisID) == 7 and oeisID[1:7].isdigit():
            boxOf_f_astConstant: list[ast.expr] = []
            NodeTourist(Be.MatchValue, Grab.valueAttribute(Then.appendTo(boxOf_f_astConstant))).visit(FunctionDef)
            if 1 == len(boxOf_f_astConstant):
                slice_ast_expr: ast.expr = boxOf_f_astConstant[0]
            else:
                slice_ast_expr = Make.Tuple(boxOf_f_astConstant)

            boxOf_arg: list[ast.expr] = unjoinBinOP(FunctionDef.args.args[1], ast.BitOr)
            boxOf_arg.reverse()
            boxOf_arg.insert(0, Make.Subscript(Make.Name('Literal'), slice=slice_ast_expr))
            NodeChanger(Be.arg.argIs('f'.__eq__), Grab.annotationAttribute(Then.replaceWith(Make.BitOr.join(boxOf_arg)))).visit(FunctionDef)

            boxOf_f: list[str] = list(map(ast.literal_eval, boxOf_f_astConstant))
            functionOf: str = ' or '.join(boxOf_f)
            metadata: MetadataOEISid = getMetadata(oeisID)

            ImaDocstring: str = f"""
    Compute {oeisID}(n) as a function of {functionOf}.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of {oeisID} is: "{metadata['description']}"

    The domain of {oeisID} starts at {metadata['offset']}, therefore for values of `n` < {metadata['offset']}, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is {metadata['valueUnknown']}.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        {metadata['description']}

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/{oeisID}
    """

            astExprDocstring: ast.Expr = Make.Expr(Make.Constant(ImaDocstring))

            NodeChanger(IfThis.isFunctionDefIdentifier(oeisID)
                , Grab.bodyAttribute(Grab.index(0, Then.insertThisAbove([astExprDocstring])))
            ).visit(astModule)

    ast.fix_missing_locations(astModule)

    docstringModule: str = raiseIfNone(ast.get_docstring(astModule))
    moduleAsString: str = ast.unparse(astModule)
    moduleAsString = moduleAsString.replace(docstringModule, docstringModule + "\n\n" + moduleWarning)

    return writePython(moduleAsString, pathFilenameWrite)
