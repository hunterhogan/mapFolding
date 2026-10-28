"""I use this module to carry datatype-import replacement settings.

(AI generated docstring)

This module defines the `Settings形` tuple exported by `mapFolding.kitAST`.
Job generation uses these records to replace abstract datatype imports.

Contents
--------
Classes
	Settings形
		Specify a replacement import for an abstract datatype.
"""
from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
	from astToolkit import identifierDotAttribute

class Settings形(NamedTuple):
	"""Specify a replacement import for an abstract datatype.

	(AI generated docstring)

	You can use this tuple to map a datatype identifier in generated source to an imported type.
	The configuration records the original identifier, replacement module, type, and import alias.
	`mapFolding.kitAST.setDatatypeViaImport` applies the configuration.

	Attributes
	----------
	datatypeIdentifier : str
		Original abstract datatype identifier whose import alias will be replaced.
	typeModule : identifierDotAttribute
		Dotted module path supplying the replacement datatype.
	typeIdentifier : str
		Datatype identifier imported from the replacement module.
	type_asname : str | None = None
		Optional alias written on the replacement import.
	"""

	datatypeIdentifier: str
	"""Inspect the original abstract datatype identifier whose import alias will be replaced.

	(AI generated docstring)
	"""
	typeModule: identifierDotAttribute
	"""Inspect the dotted module path supplying the replacement datatype.

	(AI generated docstring)
	"""
	typeIdentifier: str
	"""Inspect the datatype identifier imported from the replacement module.

	(AI generated docstring)
	"""
	type_asname: str | None = None
	"""Inspect the optional alias written on the replacement import.

	(AI generated docstring)
	"""
