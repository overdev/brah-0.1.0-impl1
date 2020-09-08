__all__ = [
    'EK_LEXR',
    'EK_SNTX',
    'EK_DECL',
    'EK_TYPE',
    'EK_STMT',
    'EK_EXPR',
    'EK_SCOP',

    # Lexer
    'ERR_UNEXP_CHAR',
    'ERR_OP_TOO_LONG',

    # Syntax
    'ERR_UNEXP_TOKEN',

    # Expression
    'ERR_INVALID_SUFFIX',
    'ERR_OUT_OF_RNG',
    'ERR_NOT_CONSTEXPR',
    'ERR_CONSTEXPR_CALL',
    'ERR_CONSTEXPR_TERNARY',
    'ERR_NOT_CALLABLE',
    'ERR_LITERAL_INCR_DECR',
    'ERR_CANNOT_INCR_DECR',
    'ERR_NOT_SIGNED',
    'ERR_DIVISION_BY_ZERO',

    # Declaration
    'ERR_UNDEFINED_NAME',
    'ERR_REDECLARED_NAME',
    'ERR_REDECLARED_LABEL',
    'ERR_VOID_RETURN',
    'ERR_NO_VOID_RETURN',

    # Statement
    'ERR_WRONG_ARG_NUMBER',
    'ERR_INVALID_STMT',

    # Type
    'ERR_WRONG_RETURN_TYPE',
    'ERR_WRONG_ARG_TYPE',
    'ERR_INCOMPATIBLE_TYPES',

    # Scope
    'ERR_WRONG_DECL_SCOPE',
    'ERR_WRONG_STMT_SCOPE',
]

# ---------------------------------------------------------
# region CONSTANTS & ENUMS

EK_LEXR = "LEXER"
EK_SNTX = "SYNTAX"
EK_DECL = "DECLARATION"
EK_TYPE = "TYPE"
EK_STMT = "STATEMENT"
EK_EXPR = "EXPRESSION"
EK_SCOP = "SCOPE"

# Lexer Errors
ERR_UNEXP_CHAR = "Unexpected char '{0}'."
ERR_OP_TOO_LONG = "Operator is too long."

# Syntax Errors
ERR_UNEXP_TOKEN = 'Expected {0}, got \'{1}\'.'

# Expression Errors
ERR_INVALID_SUFFIX = "Invalid literal suffix '{0}'."
ERR_OUT_OF_RNG = "Value out of range."
ERR_NOT_CALLABLE = "{0} cannot be called."
ERR_NOT_CONSTEXPR = "{0} is not a constant symbol."
ERR_CONSTEXPR_CALL = "Constant expressions cannot have function calls."
ERR_CONSTEXPR_TERNARY = "Constant expressions cannot have conditionals."
ERR_LITERAL_INCR_DECR = "{0} a literal value is not OK."
ERR_CANNOT_INCR_DECR = "{0} a {1} is not possible."
ERR_NOT_SIGNED = "{0} does not have sign."
ERR_DIVISION_BY_ZERO = "Division by zero in expression."

# Declaration Errors
ERR_UNDEFINED_NAME = "Undefined name '{0}'"
ERR_REDECLARED_NAME = "'{0}' name is already declared."
ERR_REDECLARED_LABEL = "'{0}' label already declared in this definition."
ERR_VOID_RETURN = "'{0}' function has a return value of type {1}."
ERR_NO_VOID_RETURN = "'{0}' function has no return value."

# Statement Errors
ERR_WRONG_ARG_NUMBER = "'{0}' function expects {1} arguments, but {2} were passed."
ERR_INVALID_STMT = "Invalid statement."

# Type Errors
ERR_WRONG_RETURN_TYPE = "'{0}' function return is of type {1}."
ERR_WRONG_ARG_TYPE = "Argument {0} of '{1}' expects {2} value but received a {3}."
ERR_INCOMPATIBLE_TYPES = "Incompatible operand types {0} in {1} expression."

# Scope Errors
ERR_WRONG_DECL_SCOPE = "{0} must be declared in module level."
ERR_WRONG_STMT_SCOPE = "'{0}' statement is ouside of {1}."

# endregion (constants)
# ---------------------------------------------------------
# region FUNCTIONS

# endregion (functions)
# ---------------------------------------------------------
# region CLASSES

# endregion (classes)
# ---------------------------------------------------------
