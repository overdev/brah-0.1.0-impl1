# from enum import Enum, auto

__all__ = [
    # 'TokenKind',
    # 'TK_UNEXPECTED',
    # 'TK_NONE',
    # 'TK_EOF',
    # 'TK_LITERAL_INT',
    # 'TK_LITERAL_FLOAT',
    # 'TK_OP_PLUS',
    # 'TK_OP_MINUS',
    # 'TK_OP_MULT',
    # 'TK_OP_DIV',
    # 'TK_OP_MOD',
    # 'TK_UNAOP_POS',
    # 'TK_UNAOP_NEG',
    # 'TK_LPAREN',
    # 'TK_RPAREN',

    'SW_MAINFUNCTION',

    'KW_VARIABLE',
    'KW_IF',
    'KW_ELSE',
    'KW_PRINT',
    'KW_FUNCTION',
    'KW_RETURN',
    'KW_AND',
    'KW_OR',
    'KW_XOR',
    'KW_NOT',
    'KW_IS',

    'TT_NAME',
    'TT_KW',
    'TT_INT',
    'TT_FLOAT',
    'TT_STR',
    'TT_LPAREN',
    'TT_RPAREN',
    'TT_LBRACE',
    'TT_RBRACE',
    'TT_LBRACKET',
    'TT_RBRACKET',
    'TT_PLUS',
    'TT_MINUS',
    'TT_MULT',
    'TT_DIV',
    'TT_MOD',
    'TT_COMMA',
    'TT_SEMI',
    'TT_COLON',
    'TT_DOT',
    'TT_EQUAL',
    'TT_QUESTION',

    'OP_EQ',
    'OP_NE',
    'OP_GT',
    'OP_GE',
    'OP_LT',
    'OP_LE',
    'OP_AND',
    'OP_OR',
    'OP_XOR',
    'OP_NOT',
    'OP_ASN',
    'OP_ADD',
    'OP_SUB',
    'OP_DIV',
    'OP_MUL',
    'OP_MOD',
    'OP_TER',
]

# ---------------------------------------------------------
# region CONSTANTS & ENUMS

SW_MAINFUNCTION = 'principal'

KW_VARIABLE = "var"
KW_IF = "se"
KW_ELSE = "senão"
KW_PRINT = "escreva"
KW_FUNCTION = "função"
KW_RETURN = "retorne"
KW_AND = "e"
KW_OR = "ou"
KW_XOR = "oux"
KW_NOT = "not"
KW_IS = "é"

TT_NAME = "NAME"
TT_KW = "KEYWORD"
TT_INT = "INTEGER"
TT_FLOAT = "FLOAT"
TT_STR = "STRING"
TT_LPAREN = "("
TT_RPAREN = ")"
TT_LBRACE = "{"
TT_RBRACE = "}"
TT_LBRACKET = "["
TT_RBRACKET = "]"
TT_PLUS = "+"
TT_MINUS = "-"
TT_MULT = "*"
TT_DIV = "/"
TT_MOD = "%"
TT_COMMA = ","
TT_SEMI = ";"
TT_COLON = ":"
TT_DOT = "."
TT_EQUAL = '='
TT_QUESTION = '?'

OP_EQ = "=="
OP_NE = "!="
OP_GT = ">"
OP_GE = ">="
OP_LT = "<"
OP_LE = "<="
OP_AND = "&"
OP_OR = "|"
OP_XOR = "^"
OP_NOT = "~"
OP_ASN = "="
OP_ADD = "+"
OP_SUB = "-"
OP_DIV = "/"
OP_MUL = "*"
OP_MOD = "%"
OP_TER = "?"


#
# class TokenKind(Enum):
#     UNEXPECTED = auto()
#     NONE = auto()
#     EOF = auto()
#     LITERAL_INT = auto()
#     LITERAL_FLOAT = auto()
#     OP_PLUS = auto()
#     OP_MINUS = auto()
#     OP_MULT = auto()
#     OP_DIV = auto()
#     OP_MOD = auto()
#     UNAOP_POS = auto()
#     UNAOP_NEG = auto()
#     LPAREN = auto()
#     RPAREN = auto()
#
#
# TK_UNEXPECTED = TokenKind.UNEXPECTED
# TK_NONE = TokenKind.NONE
# TK_EOF = TokenKind.EOF
# TK_LITERAL_INT = TokenKind.LITERAL_INT
# TK_LITERAL_FLOAT = TokenKind.LITERAL_FLOAT
# TK_OP_PLUS = TokenKind.OP_PLUS
# TK_OP_MINUS = TokenKind.OP_MINUS
# TK_OP_MULT = TokenKind.OP_MULT
# TK_OP_DIV = TokenKind.OP_DIV
# TK_OP_MOD = TokenKind.OP_MOD
# TK_UNAOP_POS = TokenKind.UNAOP_POS
# TK_UNAOP_NEG = TokenKind.UNAOP_NEG
# TK_LPAREN = TokenKind.LPAREN
# TK_RPAREN = TokenKind.RPAREN

# def export(prefix: str, enumeration: Type[Enum]):
#     globals()[enumeration.__name__] = enumeration
#     for name in enumeration.__members__:
#         export_name = f'{prefix}_{name}'
#         globals()[export_name] = enumeration.__members__[name]
#         __all__.append(export_name)


# endregion (constants)
# ---------------------------------------------------------
# region FUNCTIONS

# endregion (functions)
# ---------------------------------------------------------
# region CLASSES

# endregion (classes)
# ---------------------------------------------------------

# export('TT', TokenType)
# print(TK_NONE)
