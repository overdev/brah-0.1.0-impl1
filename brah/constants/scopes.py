from enum import IntEnum, auto

__all__ = [
    'ScopeKind',
    'SK_MODULE',
    'SK_STATEMENT',
    'SK_FUNCTION',
]

# ---------------------------------------------------------
# region CONSTANTS & ENUMS


class ScopeKind(IntEnum):
    MODULE = 0
    STATEMENT = auto()
    FUNCTION = auto()


SK_MODULE = ScopeKind.MODULE
SK_STATEMENT = ScopeKind.STATEMENT
SK_FUNCTION = ScopeKind.FUNCTION


# endregion (constants)
# ---------------------------------------------------------
# region FUNCTIONS

# endregion (functions)
# ---------------------------------------------------------
# region CLASSES

# endregion (classes)
# ---------------------------------------------------------
