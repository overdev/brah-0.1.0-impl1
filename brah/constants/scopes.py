from enum import IntEnum, auto

__all__ = [
    'ScopeKind',
    'SK_MODULE',
    'SK_STATEMENT',
    'SK_FUNCTION',
    'SK_LOOP',
    'SK_CASE',
]

# ---------------------------------------------------------
# region CONSTANTS & ENUMS


class ScopeKind(IntEnum):
    MODULE = 0
    FUNCTION = auto()
    STATEMENT = auto()
    LOOP = auto()
    CASE = auto()


SK_MODULE = ScopeKind.MODULE
SK_FUNCTION = ScopeKind.FUNCTION
SK_STATEMENT = ScopeKind.STATEMENT
SK_LOOP = ScopeKind.LOOP
SK_CASE = ScopeKind.CASE

# endregion (constants)
# ---------------------------------------------------------
# region FUNCTIONS

# endregion (functions)
# ---------------------------------------------------------
# region CLASSES

# endregion (classes)
# ---------------------------------------------------------
