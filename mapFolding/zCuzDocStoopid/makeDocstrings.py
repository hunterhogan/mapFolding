"""Make docstrings."""
from astToolkit import Grab, IfThis, Make, NodeChanger, parsePathFilename2astModule, Then
from astToolkit.transformationTools import makeDictionaryFunctionDef
from hunterMakesPy import raiseIfNone, writeStringToHere
from mapFolding import dictionaryOEIS, dictionaryOEISMapFolding, packageSettings
from pathlib import Path
import ast

# ----------------- General Settings ----------------------------------------------------------------------------------
sourcePrefix: str = 'zCuzDocStoopid'

moduleWarning = "NOTE: This is a generated file; edit the source file."

def transformOEISidByFormula(pathFilenameSource: Path) -> None:
    """Transform the docstrings of functions corresponding to OEIS sequences."""
    pathFilenameWrite: Path = pathFilenameSource.with_stem(pathFilenameSource.stem.removeprefix(sourcePrefix))
    astModule: ast.Module = parsePathFilename2astModule(pathFilenameSource)
    dictionaryFunctionDef: dict[str, ast.FunctionDef] = makeDictionaryFunctionDef(astModule)

    oeisID = 'Error during transformation' # `ast.FunctionDef.name` of function in `pathFilenameSource`.
    functionOf: str = 'Error during transformation' # The value of `functionOf` is in the docstring of function `oeisID` in `pathFilenameSource`.

    for oeisID, FunctionDef in dictionaryFunctionDef.items():
        if not oeisID.startswith('A') or not oeisID[1:7].isdigit():
            continue
        dictionary = dictionaryOEISMapFolding if oeisID in dictionaryOEISMapFolding else dictionaryOEIS
        functionOf = raiseIfNone(ast.get_docstring(FunctionDef))

        ImaDocstring= 	f"""
    Compute {oeisID}(n) as a function of {functionOf}.

    *The On-Line Encyclopedia of Integer Sequences* (OEIS) description of {oeisID} is: "{dictionary[oeisID]['description']}"

    The domain of {oeisID} starts at {dictionary[oeisID]['offset']}, therefore for values of `n` < {dictionary[oeisID]['offset']}, a(n) is undefined. The smallest value of n for which a(n)
    has not yet been computed is {dictionary[oeisID]['valueUnknown']}.

    Parameters
    ----------
    n : int
        Index (n-dex) for a(n) in the sequence of values. "n" (lower case) and "a(n)" are conventions in mathematics.

    Returns
    -------
    a(n) : int
        {dictionary[oeisID]['description']}

    Would You Like to Know More?
    ----------------------------
    OEIS : webpage
        https://oeis.org/{oeisID}
    """

        astExprDocstring = Make.Expr(Make.Constant(ImaDocstring))

        NodeChanger(
            findThis = IfThis.isFunctionDefIdentifier(oeisID)
            , doThat = Grab.bodyAttribute(Grab.index(0, Then.replaceWith(astExprDocstring)))
        ).visit(astModule)

    ast.fix_missing_locations(astModule)

    docstringModule = raiseIfNone(ast.get_docstring(astModule))
    moduleAsString = ast.unparse(astModule) + "\n"
    moduleAsString = moduleAsString.replace(docstringModule,docstringModule + "\n\n" + moduleWarning)

    writeStringToHere(moduleAsString, pathFilenameWrite)

pathRoot: Path = packageSettings.pathPackage / "algorithms"
pathFilenameSource: Path = next(iter(pathRoot.glob(f"{sourcePrefix}*.py"))).absolute()
transformOEISidByFormula(pathFilenameSource)
