# ======= Semantic replacements for ambiguous values =======

# Many values and options at https://gist.github.com/JBlond/2fea43a3049b38287e5e9cefc87b2124
asciiColorReset: str = '\33[0m'

asciiColorBlack: str = '\33[30m'
asciiColorBlue: str = '\33[34m'
asciiColorCyan: str = '\33[36m'
asciiColorGreen: str = '\33[32m'
asciiColorMagenta: str = '\33[35m'
asciiColorRed: str = '\33[31m'
asciiColorWhite: str = '\33[37m'
asciiColorYellow: str = '\33[33m'

asciiColorBackgroundBlack: str = '\33[40m'
asciiColorBackgroundBlue: str = '\33[44m'
asciiColorBackgroundCyan: str = '\33[46m'
asciiColorBackgroundGreen: str = '\33[42m'
asciiColorBackgroundMagenta: str = '\33[45m'
asciiColorBackgroundRed: str = '\33[41m'
asciiColorBackgroundWhite: str = '\33[47m'
asciiColorBackgroundYellow: str = '\33[43m'

decreasing: int = -1
"""Adjust the value due to Python syntax."""
inclusive: int = 1
"""Include the last value in a `range`: change from [p, q) to [p, q]."""
