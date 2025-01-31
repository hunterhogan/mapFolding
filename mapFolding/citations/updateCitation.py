from cffconvert.cli.create_citation import create_citation
from typing import Any, Dict
import cffconvert
import pathlib
import tomli
import inspect
import json

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
keywords: pathFilenamePackageSSOT
license: pathFilenamePackageSSOT
license-url: pathFilenamePackageSSOT
message: pathFilenameCitationSSOT
preferred-citation: pathFilenameCitationSSOT
references: to be determined
repository: pathFilenamePackageSSOT
repository-artifact: (https://pypi.org/pypi/{package_name}/json').json()['releases']
repository-code: workflows['Make GitHub Release']
title: pathFilenamePackageSSOT
type: pathFilenameCitationSSOT
url: pathFilenamePackageSSOT
version: pathFilenamePackageSSOT
"""
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

tomlPackageData: Dict[str, Any] = tomli.loads(pathFilenamePackageSSOT.read_text())['project']

citationObject: cffconvert.Citation = create_citation(infile=pathFilenameCitationSSOT, url=None)

path_cffconvert = pathlib.Path(inspect.getfile(cffconvert)).parent
pathFilenameSchema = path_cffconvert / "schemas/1.2.0/schema.json"
scheme: Dict[str, Any] = json.loads(pathFilenameSchema.read_text())
schemaSpecifications: Dict[str, Any] = scheme['properties']

for property, subProperties in schemaSpecifications.items():
    print(property, subProperties.get('items', None))
