from cffconvert.cli.create_citation import create_citation
from cffconvert.cli.validate_or_write_output import validate_or_write_output
from packaging.metadata import Metadata as PyPAMetadata
from typing import Any, Dict, Final, List, Tuple
import attrs
import cffconvert
import inspect
import json
import packaging
import packaging.metadata
import packaging.utils
import packaging.version
import pathlib
import ruamel.yaml
import tomli

listProjectURLsTarget: List[str] = ["homepage", "license", "repository"]

"""
Writing the CFF to disk is the difficult part.
    - Store in `attrs` class CitationNexus
        - one-to-one correlation with `cffconvert.lib.cff_1_2_x.citation` class Citation_1_2_x.cffobj
    - yaml converts to `cffstr`
        - TODO with the help of NOTE ??? to format the string
        To find the answer, probably start here:
            https://github.com/citation-file-format/cff-initializer-javascript
            src/store/cff.ts ~ `cffobj`
            src/store/cffstr.ts
    - write the formatted string with any text writer (which, for me, means pathlib.Path.write_text)
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

@attrs.define
class CitationNexus:
    """cff
    "required": [
        "authors",
        "cff-version",
        "message",
        "title"
    ],
    """
    cffDASHversion: str # pathFilenameCitationSSOT
    message: str # pathFilenameCitationSSOT

    abstract: str | None = None # pathFilenameCitationSSOT
    authors: list[str] = attrs.field(factory=list) # pathFilenamePackageSSOT; pyproject.toml authors
    commit: str | None = None # workflows['Make GitHub Release']
    contact: list[str] = attrs.field(factory=list) # pathFilenamePackageSSOT; pyproject.toml maintainers
    dateDASHreleased: str | None = None # workflows['Make GitHub Release']
    doi: str | None = None # pathFilenameCitationSSOT
    identifiers: list[str] = attrs.field(factory=list) # workflows['Make GitHub Release']
    keywords: list[str] = attrs.field(factory=list) # pathFilenamePackageSSOT; packaging.metadata.Metadata.keywords
    license: str | None = None # pathFilenamePackageSSOT; packaging.metadata.Metadata.license_expression
    licenseDASHurl: str | None = None # pathFilenamePackageSSOT; packaging.metadata.Metadata.project_urls: license or pyproject.toml urls license
    preferredDASHcitation: str | None = None # pathFilenameCitationSSOT
    references: list[str] = attrs.field(factory=list) # bibtex files in pathCitationSSOT. Conversion method and timing TBD.
    repositoryDASHartifact: str | None = None # (https://pypi.org/pypi/{package_name}/json').json()['releases']
    repositoryDASHcode: str | None = None # workflows['Make GitHub Release']
    repository: str | None = None # pathFilenamePackageSSOT; packaging.metadata.Metadata.project_urls: repository
    title: str | None = None # pathFilenamePackageSSOT; packaging.metadata.Metadata.name
    type: str | None = None # pathFilenameCitationSSOT
    url: str | None = None # pathFilenamePackageSSOT; packaging.metadata.Metadata.project_urls: homepage
    version: str | None = None # pathFilenamePackageSSOT; packaging.metadata.Metadata.version

    def setInStone(self, prophet: str) -> "CitationNexus":
        if prophet == "Citation":
            pass
            # "freeze" these items
            # setattr(self.cffDASHversion, 'type', Final[str])
            # setattr(self.doi, 'type', Final[str])
            # cffDASHversion: str # pathFilenameCitationSSOT
            # message: str # pathFilenameCitationSSOT
            # abstract: str | None = None # pathFilenameCitationSSOT
            # doi: str | None = None # pathFilenameCitationSSOT
            # preferredDASHcitation: str | None = None # pathFilenameCitationSSOT
            # type: str | None = None # pathFilenameCitationSSOT
        # elif ...
        return self

def getNexusCitation(pathFilenameCitationSSOT):

    # `cffconvert.cli.create_citation.create_citation()` is PAINFULLY mundane, but a major problem
    # in the CFF ecosystem is divergence. Therefore, I will use this function so that my code
    # converges with the CFF ecosystem.
    citationObject: cffconvert.Citation = create_citation(infile=pathFilenameCitationSSOT, url=None)
    # `._parse()` is a yaml loader: use it for convergence
    cffobj: Dict[Any, Any] = citationObject._parse()

    nexusCitation = CitationNexus(
        cffDASHversion=cffobj["cff-version"],
        message=cffobj["message"],
    )

    Z0Z_list: List[attrs.Attribute] = list(attrs.fields(type(nexusCitation)))
    for Z0Z_field in Z0Z_list:
        cffobjKeyName: str = Z0Z_field.name.replace("DASH", "-")
        cffobjValue = cffobj.get(cffobjKeyName)
        if cffobjValue: # An empty list will be False
            setattr(nexusCitation, Z0Z_field.name, cffobjValue)

    nexusCitation = nexusCitation.setInStone("Citation")
    return nexusCitation

def getPypaMetadata(packageData: Dict[str, Any]) -> PyPAMetadata:
    """
    Create a PyPA metadata object (version 2.4) from packageData.
    https://packaging.python.org/en/latest/specifications/core-metadata/

    Mapping for project URLs:
      - 'homepage' and 'repository' are accepted from packageData['urls'].
    """
    dictionaryProjectURLs: Dict[str, str] = {}
    for urlName, url in packageData.get("urls", {}).items():
        urlName = urlName.lower()
        if urlName in listProjectURLsTarget:
            dictionaryProjectURLs[urlName] = url

    metadataRaw = packaging.metadata.RawMetadata(
        keywords=packageData.get("keywords", []),
        license_expression=packageData.get("license", {}).get("text", ""),
        metadata_version="2.4",
        name=packaging.utils.canonicalize_name(packageData.get("name", None), validate=True),
        project_urls=dictionaryProjectURLs,
        version=packageData.get("version", None),
    )

    metadata = PyPAMetadata().from_raw(metadataRaw)
    return metadata

def addPypaMetadata(nexusCitation: CitationNexus, metadata: PyPAMetadata) -> CitationNexus:
    if not metadata.name:
        raise ValueError("Metadata name is required.")

    nexusCitation.title = metadata.name
    if metadata.version: nexusCitation.version = str(metadata.version)
    if metadata.keywords: nexusCitation.keywords = metadata.keywords
    if metadata.license_expression: nexusCitation.license = metadata.license_expression

    Z0Z_lookup: Dict[str, str] = {
        "homepage": "url",
        "license": "licenseDASHurl",
        "repository": "repository",
    }
    if metadata.project_urls:
        for urlTarget in listProjectURLsTarget:
            url = metadata.project_urls.get(urlTarget, None)
            if url:
                setattr(nexusCitation, Z0Z_lookup[urlTarget], url)

    return nexusCitation

def logistics():
    # Prefer reliable, dynamic values over hardcoded ones
    packageNameHARDCODED: str = 'mapFolding'

    packageName: str = packageNameHARDCODED
    pathRepoRoot = pathlib.Path(__file__).parent.parent.parent
    pathFilenamePackageSSOT = pathRepoRoot / 'pyproject.toml'
    filenameGitHubAction = 'updateCitation.yml'
    pathFilenameGitHubAction = pathRepoRoot / '.github' / 'workflows' / filenameGitHubAction

    filenameCitationDOTcff = 'CITATION.cff'
    pathCitations = pathRepoRoot / packageName / 'citations'
    pathFilenameCitationSSOT = pathCitations / filenameCitationDOTcff
    pathFilenameCitationDOTcffRepo = pathRepoRoot / filenameCitationDOTcff

    nexusCitation = getNexusCitation(pathFilenameCitationSSOT)

    tomlPackageData: Dict[str, Any] = tomli.loads(pathFilenamePackageSSOT.read_text())['project']
    # https://packaging.python.org/en/latest/specifications/pyproject-toml/
    pypaMetadata: PyPAMetadata = getPypaMetadata(tomlPackageData)

    nexusCitation = addPypaMetadata(nexusCitation, pypaMetadata)
    print(nexusCitation)

    # print(f"{pypaMetadata.name=}, {pypaMetadata.version=}, {pypaMetadata.keywords=}, {pypaMetadata.license_expression=}, {pypaMetadata.metadata_version=}, {pypaMetadata.project_urls=}")
    # validate_or_write_output(validate_only=True, citation=citationObjectSSOT)
    # validate_or_write_output(validate_only=True, citation=citationObjectDOTcffRepo)

def chatgpt_write_citation_to_disk(citation: CitationNexus, path: pathlib.Path) -> None:
    """Convert a CitationNexus to YAML and write it to disk."""
    citation_dict = attrs.asdict(citation, filter=lambda attr, value: value is not None)

    yaml = ruamel.yaml.YAML()
    yaml.default_flow_style = False  # Pretty-print YAML with indentation
    with path.open("w", encoding="utf-8") as file:
        yaml.dump(citation_dict, file)

if __name__ == '__main__':
    logistics()

# Dissection bench
# print(f"{pypaMetadata.name=}, {pypaMetadata.version=}, {pypaMetadata.keywords=}, {pypaMetadata.license_expression=}, {pypaMetadata.project_urls=}")
# path_cffconvert = pathlib.Path(inspect.getfile(cffconvert)).parent
# pathFilenameSchema = path_cffconvert / "schemas/1.2.0/schema.json"
# scheme: Dict[str, Any] = json.loads(pathFilenameSchema.read_text())
# schemaSpecifications: Dict[str, Any] = scheme['properties']

# for property, subProperties in schemaSpecifications.items():
#     print(property, subProperties.get('items', None))
