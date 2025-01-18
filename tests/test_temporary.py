from tests.conftest import *
from typing import Dict, List, Tuple
import importlib
import pytest

@pytest.fixture(scope="session", autouse=True)
def runSecondSetAfterAll(request: pytest.FixtureRequest):
    """Run after all other tests complete."""
    def toggleAndRerun():
        import mapFolding.importSelector
        mapFolding.importSelector.useLovelace = not mapFolding.importSelector.useLovelace
        importlib.reload(mapFolding.importSelector)

    request.addfinalizer(toggleAndRerun)

@pytest.mark.order(after="runSecondSetAfterAll")
def test_myabilitytodealwithbs(oeisID: str):
    for n in settingsOEIS[oeisID]['valuesTestValidation']:
        standardComparison(settingsOEIS[oeisID]['valuesKnown'][n], oeisIDfor_n, oeisID, n)

@pytest.mark.order(after="runSecondSetAfterAll")
def test_eff_em_el(listDimensionsTest_countFolds: List[int], foldsTotalKnown: Dict[Tuple[int, ...], int]) -> None:
    standardComparison(foldsTotalKnown[tuple(listDimensionsTest_countFolds)], countFolds, listDimensionsTest_countFolds, None, 'maximum')
