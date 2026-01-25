from typing import NamedTuple

#======== Semantic replacements for ambiguous values =======

decreasing: int = -1
"""Adjust the value due to Python syntax."""
inclusive: int = 1
"""Include the last value in a `range`: change from [p, q) to [p, q]."""
zeroIndexed: int = 1
"""Adjust the value due to Python syntax."""

#-------- Some color for printing text to the terminal -------
# Many values and options at https://gist.github.com/JBlond/2fea43a3049b38287e5e9cefc87b2124
# Many, many, many at https://stackoverflow.com/a/33206814/4403878
# NOTE Always define color and background color at the same time.

ansiColorReset: str = '\x1b[0m'

class AnsiColors(NamedTuple):
	BlackOnCyan: str = '\x1b[30;46m'
	BlackOnMagenta: str = '\x1b[30;45m'
	BlackOnWhite: str = '\x1b[30;47m'
	BlackOnYellow: str = '\x1b[30;43m'
	BlueOnWhite: str = '\x1b[34;47m'
	BlueOnYellow: str = '\x1b[34;43m'
	CyanOnBlack: str = '\x1b[36;40m'
	CyanOnBlue: str = '\x1b[36;44m'
	CyanOnMagenta: str = '\x1b[36;45m'
	GreenOnBlack: str = '\x1b[32;40m'
	MagentaOnBlack: str = '\x1b[35;40m'
	MagentaOnBlue: str = '\x1b[35;44m'
	MagentaOnCyan: str = '\x1b[35;46m'
	RedOnWhite: str = '\x1b[31;47m'
	WhiteOnBlack: str = '\x1b[37;40m'
	WhiteOnBlue: str = '\x1b[37;44m'
	WhiteOnMagenta: str = '\x1b[37;45m'
	WhiteOnRed: str = '\x1b[37;41m'
	YellowOnBlack: str = '\x1b[33;40m'
	YellowOnBlue: str = '\x1b[33;44m'
	YellowOnRed: str = '\x1b[33;41m'

ansiColors = AnsiColors()
