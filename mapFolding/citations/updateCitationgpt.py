from cffconvert.cli.create_citation import create_citation
from cffconvert.cli.validate_or_write_output import validate_or_write_output
from typing import Any, Dict
import cffconvert
import pathlib
import packaging.metadata
import tomli
import ruamel.yaml
import packaging
from packaging.metadata import Metadata as PyPAMetadata
import packaging.utils
import packaging.version

def addPypaMetadata(citation: cffconvert.Citation, metadata: PyPAMetadata) -> cffconvert.Citation:
    """
    Map the PyPA metadata to the citation's internal representation.

    Mapping:
      - title: metadata.name
      - version: metadata.version (converted to string)
      - keywords: metadata.keywords
      - license: metadata.license_expression
      - url: from project URLs (homepage)
      - repository: from project URLs (repository)
    """
    # Access the internal dictionary (used for conversion)
    citationData: Dict[str, Any] = citation._cffobj

    # Update title from PyPA metadata name
    if metadata.name:
        citationData["title"] = metadata.name

    # Update version from PyPA metadata version
    if metadata.version:
        citationData["version"] = str(metadata.version)

    # Update keywords from PyPA metadata keywords
    if metadata.keywords:
        citationData["keywords"] = metadata.keywords

    # Update license from PyPA metadata license_expression
    if metadata.license_expression:
        citationData["license"] = metadata.license_expression

    # Retrieve the project URLs that were attached in getPypaMetadata
    projectURLs: Dict[str, str] = getattr(metadata, "_project_urls", {})

    # Update the homepage URL
    if "homepage" in projectURLs:
        citationData["url"] = projectURLs["homepage"]

    # Update the repository URL
    if "repository" in projectURLs:
        citationData["repository"] = projectURLs["repository"]

    return citation

def getPypaMetadata(packageData: Dict[str, Any]) -> PyPAMetadata:
    """
    Create a PyPA metadata object (version 2.4) from packageData.

    Mapping for project URLs:
      - 'homepage' and 'repository' are accepted from packageData['urls'].
    """
    dictionaryProjectURLs: Dict[str, str] = {}
    for urlKey, urlValue in packageData.get("urls", {}).items():
        lowerUrlKey = urlKey.lower()
        if lowerUrlKey in ("homepage", "repository"):
            dictionaryProjectURLs[lowerUrlKey] = urlValue

    metadataRaw = packaging.metadata.RawMetadata(
        keywords=packageData.get("keywords", []),
        license_expression=packageData.get("license", {}).get("text", ""),
        metadata_version="2.4",
        name=packaging.utils.canonicalize_name(packageData.get("name", None), validate=True),
        project_urls=dictionaryProjectURLs,
        version=packageData.get("version", None),
    )

    metadata = PyPAMetadata().from_raw(metadataRaw)
    # Attach the project URLs dictionary so it can be used later.
    setattr(metadata, "_project_urls", dictionaryProjectURLs)
    return metadata

def logistics():
    # Determine paths from your SSOT.
    packageName: str = "mapFolding"
    pathRepoRoot = pathlib.Path(__file__).parent.parent.parent
    pathFilenamePackageSSOT = pathRepoRoot / "pyproject.toml"
    filenameGitHubAction = "updateCitation.yml"
    pathFilenameGitHubAction = pathRepoRoot / ".github" / "workflows" / filenameGitHubAction

    filenameCitationDOTcff = "CITATION.cff"
    pathCitations = pathRepoRoot / packageName / "citations"
    pathFilenameCitationSSOT = pathCitations / filenameCitationDOTcff
    pathFilenameCitationDOTcffRepo = pathRepoRoot / filenameCitationDOTcff

    # Create a citation object from the SSOT citation file.
    citationObject: cffconvert.Citation = create_citation(infile=pathFilenameCitationSSOT, url=None)
    # Print the current citation in CFF format (for debugging) using the as_cff method.
    print(citationObject.as_cff())

    # Load package metadata from pyproject.toml.
    tomlPackageData: Dict[str, Any] = tomli.loads(pathFilenamePackageSSOT.read_text())["project"]
    pypaMetadata: PyPAMetadata = getPypaMetadata(tomlPackageData)

    # Map the PyPA metadata into the citation's internal representation.
    citationObject = addPypaMetadata(citation=citationObject, metadata=pypaMetadata)

    # Validate and write out the updated citation file in both locations.
    # validate_or_write_output(
    #     outfile=pathFilenameCitationSSOT,
    #     outputformat="cff",
    #     validate_only=False,
    #     citation=citationObject,
    # )
    validate_or_write_output(
        outfile=pathFilenameCitationDOTcffRepo,
        outputformat="cff",
        validate_only=False,
        citation=citationObject,
    )

if __name__ == "__main__":
    logistics()
