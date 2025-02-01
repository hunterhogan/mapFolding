from cffconvert.cli.create_citation import create_citation
from cffconvert.cli.validate_or_write_output import validate_or_write_output
from typing import Any, Dict
import cffconvert
import pathlib
import packaging.metadata
import tomli
import inspect
import json
import ruamel.yaml
import packaging
from packaging.metadata import Metadata as PyPAMetadata
import packaging.utils
import packaging.version

def addPypaMetadata(citation: cffconvert.Citation, metadata: PyPAMetadata) -> cffconvert.Citation:
    # https://github.com/citation-file-format/cff-initializer-javascript
    """
keywords: pathFilenamePackageSSOT; packaging.metadata.Metadata.keywords
license: pathFilenamePackageSSOT; packaging.metadata.Metadata.license_expression
title: pathFilenamePackageSSOT; packaging.metadata.Metadata.name
url: pathFilenamePackageSSOT; packaging.metadata.Metadata.project_urls: homepage
repository: pathFilenamePackageSSOT; packaging.metadata.Metadata.project_urls: repository
version: pathFilenamePackageSSOT; packaging.metadata.Metadata.version
    """
    return citation

def getPypaMetadata(packageData: Dict[str, Any]) -> PyPAMetadata:
    # https://packaging.python.org/en/latest/specifications/core-metadata/
    dictionaryProjectURLs = {}
    for urlKey, urlValue in packageData.get('urls', {}).items():
        if urlKey.lower() in ('homepage', 'repository'):
            dictionaryProjectURLs[urlKey] = urlValue

    metadataRaw = packaging.metadata.RawMetadata(
        keywords=packageData.get('keywords', []),
        license_expression=packageData.get('license', {}).get('text', ''),
        metadata_version='2.4',
        name=packaging.utils.canonicalize_name(packageData.get('name', None), validate=True),
        project_urls=dictionaryProjectURLs,
        version=packageData.get('version', None),
    )

    metadata = PyPAMetadata().from_raw(metadataRaw)
    return metadata

"""
Tentative plan:
- Commit and push to GitHub
- GitHub Action gathers information from the sources of truth
- If the citation needs to be updated, write to both
    - pathFilenameCitationSSOT
    - pathFilenameCitationDOTcffRepo
- Commit and push to GitHub
    - this complicates things
    - I want the updated citation to be in the `commit` field of itself
"""
"""cffconvert.Citation fields and the source of truth
abstract: pathFilenameCitationSSOT
authors: pathFilenamePackageSSOT
cff-version: pathFilenameCitationSSOT
commit: workflows['Make GitHub Release']
contact: pathFilenamePackageSSOT
date-released: workflows['Make GitHub Release']
doi: pathFilenameCitationSSOT
identifiers: workflows['Make GitHub Release']
license-url: pathFilenamePackageSSOT
message: pathFilenameCitationSSOT
preferred-citation: pathFilenameCitationSSOT
references: to be determined
repository-artifact: (https://pypi.org/pypi/{package_name}/json').json()['releases']
repository-code: workflows['Make GitHub Release']
type: pathFilenameCitationSSOT

keywords: pathFilenamePackageSSOT; packaging.metadata.Metadata.keywords
license: pathFilenamePackageSSOT; packaging.metadata.Metadata.license_expression
title: pathFilenamePackageSSOT; packaging.metadata.Metadata.name
url: pathFilenamePackageSSOT; packaging.metadata.Metadata.project_urls: homepage
repository: pathFilenamePackageSSOT; packaging.metadata.Metadata.project_urls: repository
version: pathFilenamePackageSSOT; packaging.metadata.Metadata.version
"""

def logistics():
    # Prefer reliable, dynamic values over hardcoded ones
    packageName: str = 'mapFolding'
    pathRepoRoot = pathlib.Path(__file__).parent.parent.parent
    pathFilenamePackageSSOT = pathRepoRoot / 'pyproject.toml'
    filenameGitHubAction = 'updateCitation.yml'
    pathFilenameGitHubAction = pathRepoRoot / '.github' / 'workflows' / filenameGitHubAction

    filenameCitationDOTcff = 'CITATION.cff'
    pathCitations = pathRepoRoot / packageName / 'citations'
    pathFilenameCitationSSOT = pathCitations / filenameCitationDOTcff
    pathFilenameCitationDOTcffRepo = pathRepoRoot / filenameCitationDOTcff

    citationObject: cffconvert.Citation = create_citation(infile=pathFilenameCitationSSOT, url=None)
    print(citationObject._parse().as_cff())

    tomlPackageData: Dict[str, Any] = tomli.loads(pathFilenamePackageSSOT.read_text())['project']
    # https://packaging.python.org/en/latest/specifications/pyproject-toml/
    pypaMetadata: PyPAMetadata = getPypaMetadata(tomlPackageData)

    validate_or_write_output(outfile=pathFilenameCitationSSOT, outputformat='cff', validate_only=False, citation=citationObject)
    validate_or_write_output(outfile=pathFilenameCitationDOTcffRepo, outputformat='cff', validate_only=False, citation=citationObject)

if __name__ == '__main__':
    logistics()

    # print(f"{pypaMetadata.name=}, {pypaMetadata.version=}, {pypaMetadata.keywords=}, {pypaMetadata.license_expression=}, {pypaMetadata.project_urls=}")
# path_cffconvert = pathlib.Path(inspect.getfile(cffconvert)).parent
# pathFilenameSchema = path_cffconvert / "schemas/1.2.0/schema.json"
# scheme: Dict[str, Any] = json.loads(pathFilenameSchema.read_text())
# schemaSpecifications: Dict[str, Any] = scheme['properties']

# for property, subProperties in schemaSpecifications.items():
#     print(property, subProperties.get('items', None))
