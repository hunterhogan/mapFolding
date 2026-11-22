from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from typing import LiteralString

type OEISid = LiteralString
"""Identify an integer sequence with an OEIS label.

(AI generated docstring)

This alias uses `typing.LiteralString` for sequence identifiers such as `'A000682'`.
The annotation does not validate identifier syntax or establish that a sequence is registered
with the package.
"""
