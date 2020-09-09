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
    'KW_CONSTANT',
    'KW_ENUMERATION',
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
    'KW_WHILE',
    'KW_DO',
    'KW_UNTIL',
    'KW_FOR',
    'KW_IN',
    'KW_OF',
    'KW_REPEAT',
    'KW_CONTINUE',
    'KW_BREAK',

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
    'OP_INC',
    'OP_DEC',

    'TY_VOID',
    'TY_BOOL',
    'TY_INT8',
    'TY_INT16',
    'TY_INT32',
    'TY_INT64',
    'TY_UINT8',
    'TY_UINT16',
    'TY_UINT32',
    'TY_UINT64',
    'TY_FLOAT16',
    'TY_FLOAT32',
    'TY_FLOAT64',
    'TY_FLOAT80',

    'SFX_INT8',
    'SFX_INT16',
    'SFX_INT32',
    'SFX_INT64',
    'SFX_UINT8',
    'SFX_UINT16',
    'SFX_UINT32',
    'SFX_UINT64',
    'SFX_FLOAT16',
    'SFX_FLOAT32',
    'SFX_FLOAT64',
    'SFX_FLOAT80',

    'SUFFIXES',
    'KEYWORDS',
    'OPERATORS',
    'PRIMITIVES',
    'TYPE_MODIFIERS',

]

# ---------------------------------------------------------
# region CONSTANTS & ENUMS

SW_MAINFUNCTION = 'principal'

KW_VARIABLE = "var"
KW_CONSTANT = "constante"
KW_ENUMERATION = "enumeração"
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
KW_WHILE = "enquanto"
KW_DO = "faça"
KW_UNTIL = "até"
KW_FOR = "para"
KW_IN = "em"
KW_OF = "de"
KW_REPEAT = "repita"
KW_CONTINUE = "continue"
KW_BREAK = "pare"

KEYWORDS = [name for name in globals() if name.startswith('KW_')]

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
OP_INC = "++"
OP_DEC = "--"

OPERATORS = [name for name in globals() if name.startswith('OP_')]

TY_VOID = "vazio"
TY_BOOL = "booleano"
TY_INT8 = "int8s"
TY_INT16 = "int16s"
TY_INT32 = "int32s"
TY_INT64 = "int64s"
TY_UINT8 = "int8d"
TY_UINT16 = "int16d"
TY_UINT32 = "int32d"
TY_UINT64 = "int64d"
TY_FLOAT16 = "real16"
TY_FLOAT32 = "real32"
TY_FLOAT64 = "real64"
TY_FLOAT80 = "real80"

PRIMITIVES = [name for name in globals() if name.startswith('TY_')]

SFX_INT8 = "u"
SFX_INT16 = "c"
SFX_INT32 = "i"
SFX_INT64 = "l"
SFX_UINT8 = "ud"
SFX_UINT16 = "cd"
SFX_UINT32 = "id"
SFX_UINT64 = "ld"
SFX_FLOAT16 = "m"
SFX_FLOAT32 = "r"
SFX_FLOAT64 = "d"
SFX_FLOAT80 = "x"

SUFFIXES = [name for name in globals() if name.startswith('SFX_')]

TYMOD_ISIGNED = "sinalado"
TYMOD_IUNSIGNED = "dessinalado"

TYPE_MODIFIERS = [name for name in globals() if name.startswith('TYMOD_')]

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
