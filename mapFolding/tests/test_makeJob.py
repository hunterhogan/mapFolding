"""Generated Numba counting jobs, `mapFolding.kitAST.numba.makeJob.makeJobNumba`."""
from __future__ import annotations

from hunterMakesPy import raiseIfNone
from mapFolding.kitAST.numba.kitNumba import parametersNumbaDefault
from mapFolding.kitAST.numba.makeJob import makeJobNumba
from mapFolding.kitFilesystem import readText
from mapFolding.tests import assertEqualTo
from pathlib import Path
from typing import TYPE_CHECKING
import pytest
import runpy

if TYPE_CHECKING:
	from mapFolding.kitAST.RecipeJob import RecipeJobTheorem2
	from mapFolding.theTypes import OEISid

@pytest.mark.parametrize('useNumbaProgressBar', (False, True))
@pytest.mark.parametrize('job, oeisID, n', (pytest.param((2, 4), 'A001415', 4, id='2x4'),), indirect=['job'])
def test_makeJobNumba(job: RecipeJobTheorem2, useNumbaProgressBar: bool, oeisID: OEISid, n: int, expected: int) -> None:
	"""Running the generated module writes the folding count to the count file of `job`."""
	runpy.run_path(str(makeJobNumba(job, useNumbaProgressBar=useNumbaProgressBar, parametersNumba=parametersNumbaDefault)), run_name='__main__')
	assertEqualTo(readText(Path(raiseIfNone(job.pathFilenameTotalFolds))).strip(), str(expected), makeJobNumba.__name__, job.state.mapShape
		, useNumbaProgressBar=useNumbaProgressBar)
