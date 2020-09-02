from enum import IntEnum, auto

__all__ = [
    'NodeKind',
    'NK_NONE',
    'NK_INT8_EXPR',
    'NK_INT16_EXPR',
    'NK_INT32_EXPR',
    'NK_INT64_EXPR',
    'NK_NULL_EXPR',
    'NK_UNDEFINED_EXPR',
    'NK_FUNCTION_EXPR',
    'NK_FCALL_EXPR',
    'NK_VAR_EXPR',
    'NK_PARAM_EXPR',
    'NK_OPERAND_EXPR',
    'NK_UNARY_EXPR',
    'NK_INC_EXPR',
    'NK_DEC_EXPR',
    'NK_BINARY_EXPR',
    'NK_TERNARY_EXPR',
    'NK_LOGIC_EXPR',
    'NK_COMPARISON_EXPR',
    'NK_EXPR',
    'NK_NAME',
    'NK_FUNCTION_DECL',
    'NK_VAR_DECL',
    'NK_PARAM_DECL',
    'NK_ASSIGN_STMT',
    'NK_INC_STMT',
    'NK_DEC_STMT',
    'NK_PRINT_STMT',
    'NK_RETURN_STMT',
    'NK_IF_THEN_STMT',
    'NK_IF_ELSE_STMT',
    'NK_WHILE_STMT',
    'NK_DO_WHILE_STMT',
    'NK_DO_UNTIL_STMT',
    'NK_FOR_STMT',
    'NK_FOR_IN_STMT',
    'NK_FOR_OF_STMT',
    'NK_REPEAT_FINITE_STMT',
    'NK_REPEAT_INFINITE_STMT',
    'NK_BREAK_STMT',
    'NK_CONTINUE_STMT',
    'NK_STMT',
    'NK_BLOCK',
    'NK_MODULE_ASM',
    'NK_SHARED_ASM',
    'NK_MODULE_SCOPE',
    'NK_FUNCTION_SCOPE',
    'NK_CLASS_SCOPE',
    'NK_METHOD_SCOPE',
    'NK_STATEMENT_SCOPE',
    'NK_LOOP_SCOPE',
    'NK_CASE_SCOPE',

    'NK_SCOPES',
    'NK_STATEMENTS',
    'NK_EXPRESSIONS',
    'NK_DECLARARIONS',
]

# ---------------------------------------------------------
# region CONSTANTS & ENUMS
from typing import Tuple


class NodeKind(IntEnum):
    NONE = auto()
    INT8_EXPR = auto()
    INT16_EXPR = auto()
    INT32_EXPR = auto()
    INT64_EXPR = auto()
    NULL_EXPR = auto()
    UNDEFINED_EXPR = auto()
    FUNCTION_EXPR = auto()
    FCALL_EXPR = auto()
    VAR_EXPR = auto()
    PARAM_EXPR = auto()
    OPERAND_EXPR = auto()
    UNARY_EXPR = auto()
    INC_EXPR = auto()
    DEC_EXPR = auto()
    BINARY_EXPR = auto()
    TERNARY_EXPR = auto()
    LOGIC_EXPR = auto()
    COMPARISSON_EXPR = auto()
    EXPR = auto()
    NAME = auto()
    FUNCTION_DECL = auto()
    VAR_DECL = auto()
    PARAM_DECL = auto()
    ASSIGN_STMT = auto()
    INC_STMT = auto()
    DEC_STMT = auto()
    PRINT_STMT = auto()
    RETURN_STMT = auto()
    IF_THEN_STMT = auto()
    IF_ELSE_STMT = auto()
    WHILE_STMT = auto()
    DO_WHILE_STMT = auto()
    DO_UNTIL_STMT = auto()
    FOR_STMT = auto()
    FOR_IN_STMT = auto()
    FOR_OF_STMT = auto()
    REPEAT_FINITE_STMT = auto()
    REPEAT_INFINITE_STMT = auto()
    BREAK_STMT = auto()
    CONTINUE_STMT = auto()
    STMT = auto()
    BLOCK = auto()
    MODULE_ASM = auto()
    SHARED_ASM = auto()
    MODULE_SCOPE = auto()
    FUNCTION_SCOPE = auto()
    CLASS_SCOPE = auto()
    METHOD_SCOPE = auto()
    STATEMENT_SCOPE = auto()
    LOOP_SCOPE = auto()
    CASE_SCOPE = auto()


NK_NONE = NodeKind.NONE
NK_INT8_EXPR = NodeKind.INT8_EXPR
NK_INT16_EXPR = NodeKind.INT16_EXPR
NK_INT32_EXPR = NodeKind.INT32_EXPR
NK_INT64_EXPR = NodeKind.INT64_EXPR
NK_NULL_EXPR = NodeKind.NULL_EXPR
NK_UNDEFINED_EXPR = NodeKind.UNDEFINED_EXPR
NK_FUNCTION_EXPR = NodeKind.FUNCTION_EXPR
NK_FCALL_EXPR = NodeKind.FCALL_EXPR
NK_VAR_EXPR = NodeKind.VAR_EXPR
NK_PARAM_EXPR = NodeKind.PARAM_EXPR
NK_OPERAND_EXPR = NodeKind.OPERAND_EXPR
NK_UNARY_EXPR = NodeKind.UNARY_EXPR
NK_INC_EXPR = NodeKind.INC_EXPR
NK_DEC_EXPR = NodeKind.DEC_EXPR
NK_BINARY_EXPR = NodeKind.BINARY_EXPR
NK_TERNARY_EXPR = NodeKind.TERNARY_EXPR
NK_LOGIC_EXPR = NodeKind.LOGIC_EXPR
NK_COMPARISON_EXPR = NodeKind.COMPARISSON_EXPR
NK_EXPR = NodeKind.EXPR
NK_NAME = NodeKind.NAME
NK_FUNCTION_DECL = NodeKind.FUNCTION_DECL
NK_VAR_DECL = NodeKind.VAR_DECL
NK_PARAM_DECL = NodeKind.PARAM_DECL
NK_ASSIGN_STMT = NodeKind.ASSIGN_STMT
NK_INC_STMT = NodeKind.INC_STMT
NK_DEC_STMT = NodeKind.DEC_STMT
NK_PRINT_STMT = NodeKind.PRINT_STMT
NK_RETURN_STMT = NodeKind.RETURN_STMT
NK_IF_THEN_STMT = NodeKind.IF_THEN_STMT
NK_IF_ELSE_STMT = NodeKind.IF_ELSE_STMT
NK_WHILE_STMT = NodeKind.WHILE_STMT
NK_DO_WHILE_STMT = NodeKind.DO_WHILE_STMT
NK_DO_UNTIL_STMT = NodeKind.DO_UNTIL_STMT
NK_FOR_STMT = NodeKind.FOR_STMT
NK_FOR_IN_STMT = NodeKind.FOR_IN_STMT
NK_FOR_OF_STMT = NodeKind.FOR_OF_STMT
NK_REPEAT_FINITE_STMT = NodeKind.REPEAT_FINITE_STMT
NK_REPEAT_INFINITE_STMT = NodeKind.REPEAT_INFINITE_STMT
NK_BREAK_STMT = NodeKind.BREAK_STMT
NK_CONTINUE_STMT = NodeKind.CONTINUE_STMT
NK_STMT = NodeKind.STMT
NK_BLOCK = NodeKind.BLOCK
NK_MODULE_ASM = NodeKind.MODULE_ASM
NK_SHARED_ASM = NodeKind.SHARED_ASM
NK_MODULE_SCOPE = NodeKind.MODULE_SCOPE
NK_FUNCTION_SCOPE = NodeKind.FUNCTION_SCOPE
NK_CLASS_SCOPE = NodeKind.CLASS_SCOPE
NK_METHOD_SCOPE = NodeKind.METHOD_SCOPE
NK_STATEMENT_SCOPE = NodeKind.STATEMENT_SCOPE
NK_LOOP_SCOPE = NodeKind.LOOP_SCOPE
NK_CASE_SCOPE = NodeKind.CASE_SCOPE

NK_SCOPES: Tuple[NodeKind, ...] = tuple(nk for nk in NodeKind if nk.name.endswith('SCOPE'))
NK_STATEMENTS: Tuple[NodeKind, ...] = tuple(nk for nk in NodeKind if nk.name.endswith('STMT'))
NK_EXPRESSIONS: Tuple[NodeKind, ...] = tuple(nk for nk in NodeKind if nk.name.endswith('EXPR'))
NK_DECLARARIONS: Tuple[NodeKind, ...] = tuple(nk for nk in NodeKind if nk.name.endswith('DECL'))

# endregion (constants)
# ---------------------------------------------------------
# region FUNCTIONS

# endregion (functions)
# ---------------------------------------------------------
# region CLASSES

# endregion (classes)
# ---------------------------------------------------------
