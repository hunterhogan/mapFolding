# ======= Semantic replacements for ambiguous values =======

decreasing: int = -1
"""Adjust the value due to Python syntax."""
inclusive: int = 1
"""Include the last value in a `range`: change from [p, q) to [p, q]."""
zeroIndexed: int = 1
"""Adjust the value due to Python syntax."""

# ------- Some color for printing text to the terminal -------
# Many values and options at https://gist.github.com/JBlond/2fea43a3049b38287e5e9cefc87b2124
# Many, many, many at https://stackoverflow.com/a/33206814/4403878
# NOTE Always define color and background color at the same time.

ansiColorReset:str = '\x1b[0m'

ansiColors: list[str] = [
	ansiColorBlackOnCyan	:= '\x1b[30;46m',
	ansiColorBlackOnMagenta	:= '\x1b[30;45m',
	ansiColorBlackOnWhite	:= '\x1b[30;47m',
	ansiColorBlackOnYellow	:= '\x1b[30;43m',
	ansiColorBlueOnWhite	:= '\x1b[34;47m',
	ansiColorBlueOnYellow	:= '\x1b[34;43m',
	ansiColorCyanOnBlack	:= '\x1b[36;40m',
	ansiColorCyanOnBlue		:= '\x1b[36;44m',
	ansiColorCyanOnMagenta	:= '\x1b[36;45m',
	ansiColorGreenOnBlack	:= '\x1b[32;40m',
	ansiColorMagentaOnBlack	:= '\x1b[35;40m',
	ansiColorMagentaOnBlue	:= '\x1b[35;44m',
	ansiColorMagentaOnCyan	:= '\x1b[35;46m',
	ansiColorRedOnWhite		:= '\x1b[31;47m',
	ansiColorWhiteOnBlack	:= '\x1b[37;40m',
	ansiColorWhiteOnBlue	:= '\x1b[37;44m',
	ansiColorWhiteOnMagenta	:= '\x1b[37;45m',
	ansiColorWhiteOnRed		:= '\x1b[37;41m',
	ansiColorYellowOnBlack	:= '\x1b[33;40m',
	ansiColorYellowOnBlue	:= '\x1b[33;44m',
	ansiColorYellowOnRed	:= '\x1b[33;41m',
]
