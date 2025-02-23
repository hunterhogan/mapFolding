from numba.core.errors import NumbaWarning as NumbaWarning

def disassemble_elf_to_cfg(elf, mangled_symbol):
    """
    Gets the CFG of the disassembly of an ELF object, elf, at mangled name,
    mangled_symbol, and renders it appropriately depending on the execution
    environment (terminal/notebook).
    """
