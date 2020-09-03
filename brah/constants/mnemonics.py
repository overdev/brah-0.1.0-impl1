from enum import IntEnum, auto

__all__ = [
    'Opcode',
    'OPCODES',
    'CONST',
    'HALT',
    'NOP',
    'CONST_B',
    'CONST_W',
    'CONST_D',
    'CONST_Q',
    'CONST_SB',
    'CONST_SW',
    'CONST_SD',
    'CONST_SQ',
    'CONST_FH',
    'CONST_FS',
    'CONST_FD',
    'CONST_FX',
    'ADD',
    'SADD',
    'FADD',
    'SUB',
    'SSUB',
    'FSUB',
    'MUL',
    'SMUL',
    'FMUL',
    'DIV',
    'SDIV',
    'FDIV',
    'MOD',
    'SMOD',
    'FMOD',
    'POS',
    'NEG',
    'GET',
    'SET',
    'CALL',
    'RET',
    'JMP',
    'CMP',
    'JE',
    'JNE',
    'JA',
    'JAE',
    'JB',
    'JBE',
    'JZ',
    'JNZ',
    'AND',
    'ANDL',
    'OR',
    'ORL',
    'XOR',
    'XORL',
    'NOT',
    'NOTL',
    'LT',
    'LTE',
    'EQ',
    'NE',
    'GTE',
    'GT',
    'INC',
    'DEC',
    'IRQ',
]

# ---------------------------------------------------------
# region CONSTANTS & ENUMS


class Opcode(IntEnum):
    HALT = 0
    NOP = auto()
    CONST_B = auto()  # BYTE INTEGER
    CONST_W = auto()  # WORD INTEGER
    CONST_D = auto()  # DOUBLE WORD INTEGER
    CONST_Q = auto()  # QUAD WORD INTEGER
    CONST_SB = auto()  # UNSIGNED BYTE INTEGER
    CONST_SW = auto()  # UNSIGNED WORD INTEGER
    CONST_SD = auto()  # UNSIGNED DOUBLE WORD INTEGER
    CONST_SQ = auto()  # UNSIGNED QUAD WORD INTEGER
    CONST_FH = auto()  # HALP PRECISION FLOAT
    CONST_FS = auto()  # SINGLE PRECISION FLOAT
    CONST_FD = auto()  # DOUBLE PRECISION FLOAT
    CONST_FX = auto()  # EXTENDED PRECISION FLOAT
    ADD = auto()
    SADD = auto()
    FADD = auto()
    SUB = auto()
    SSUB = auto()
    FSUB = auto()
    MUL = auto()
    SMUL = auto()
    FMUL = auto()
    DIV = auto()
    SDIV = auto()
    FDIV = auto()
    MOD = auto()
    SMOD = auto()
    FMOD = auto()
    POS = auto()
    NEG = auto()
    GET = auto()
    SET = auto()
    CALL = auto()
    RET = auto()
    JMP = auto()
    CMP = auto()
    JE = auto()
    JNE = auto()
    JA = auto()
    JAE = auto()
    JB = auto()
    JBE = auto()
    JZ = auto()
    JNZ = auto()
    AND = auto()
    ANDL = auto()
    OR = auto()
    ORL = auto()
    XOR = auto()
    XORL = auto()
    NOT = auto()
    NOTL = auto()
    LT = auto()
    LTE = auto()
    EQ = auto()
    NE = auto()
    GTE = auto()
    GT = auto()
    INC = auto()
    DEC = auto()
    IRQ = auto()


HALT = Opcode.HALT
NOP = Opcode.NOP
CONST_B = Opcode.CONST_B
CONST_W = Opcode.CONST_W
CONST_D = Opcode.CONST_D
CONST_Q = Opcode.CONST_Q
CONST_SB = Opcode.CONST_SB
CONST_SW = Opcode.CONST_SW
CONST_SD = Opcode.CONST_SD
CONST_SQ = Opcode.CONST_SQ
CONST_FH = Opcode.CONST_FH
CONST_FS = Opcode.CONST_FS
CONST_FD = Opcode.CONST_FD
CONST_FX = Opcode.CONST_FX
ADD = Opcode.ADD
SADD = Opcode.SADD
FADD = Opcode.FADD
SUB = Opcode.SUB
SSUB = Opcode.SSUB
FSUB = Opcode.FSUB
MUL = Opcode.MUL
SMUL = Opcode.SMUL
FMUL = Opcode.FMUL
DIV = Opcode.DIV
SDIV = Opcode.SDIV
FDIV = Opcode.FDIV
MOD = Opcode.MOD
SMOD = Opcode.SMOD
FMOD = Opcode.FMOD
POS = Opcode.POS
NEG = Opcode.NEG
GET = Opcode.GET
SET = Opcode.SET
CALL = Opcode.CALL
RET = Opcode.RET
JMP = Opcode.JMP
CMP = Opcode.CMP
JE = Opcode.JE
JNE = Opcode.JNE
JA = Opcode.JA
JAE = Opcode.JAE
JB = Opcode.JB
JBE = Opcode.JBE
JZ = Opcode.JZ
JNZ = Opcode.JNZ
AND = Opcode.AND
ANDL = Opcode.ANDL
OR = Opcode.OR
ORL = Opcode.ORL
XOR = Opcode.XOR
XORL = Opcode.XORL
NOT = Opcode.NOT
NOTL = Opcode.NOTL
LT = Opcode.LT
LTE = Opcode.LTE
EQ = Opcode.EQ
NE = Opcode.NE
GTE = Opcode.GTE
GT = Opcode.GT
INC = Opcode.INC
DEC = Opcode.DEC
IRQ = Opcode.IRQ

CONST = [op for op in Opcode if op.name.startswith('CONST')]
OPCODES = [name for name in Opcode.__members__]

print(CONST)

# endregion (constants)
# ---------------------------------------------------------
# region FUNCTIONS

# endregion (functions)
# ---------------------------------------------------------
# region CLASSES

# endregion (classes)
# ---------------------------------------------------------
