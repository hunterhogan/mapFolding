from mapFolding import MetadataOEISidMeanders
from mapFolding.oeis import getOEISidInformation, getOEISidValues

oeisIDsMeanders: list[str] = [
	'A000560',
	'A000682',
	'A001010',
	'A001011',
	'A005315',
	'A005316',
	'A060206',
	'A077460',
	'A078591',
	'A223094',
	'A259702',
	'A301620',
]

dictionaryOEISMeanders: dict[str, MetadataOEISidMeanders] = {
	oeisID: {
		'description': getOEISidInformation(oeisID)[0],
		'offset': getOEISidInformation(oeisID)[1],
		'valuesKnown': getOEISidValues(oeisID),
	}
	for oeisID in oeisIDsMeanders
}

