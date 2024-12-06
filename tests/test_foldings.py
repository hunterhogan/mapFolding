from mapFolding import foldings, dimensionsFoldingsTotalLookup
import pytest

def test_foldings():
    listTestsArbitrary =[
        [2,11]        
    ]
    for dimensions in listTestsArbitrary:
        dimensions.sort()
        foldingsTotal = foldings(dimensions)
        expected_from_lookup = dimensionsFoldingsTotalLookup.get(tuple(dimensions))
        assert foldingsTotal == expected_from_lookup