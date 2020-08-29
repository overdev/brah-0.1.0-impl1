from contextlib import contextmanager
from typing import Optional, List, Dict, Union, cast, Any, Tuple
from brah.constants.nodes import *
from brah.constants.mnemonics import *
from brah.c_parser import *

__all__ = [
    'Compiler',
]

# ---------------------------------------------------------
# region CONSTANTS & ENUMS

# endregion (constants)
# ---------------------------------------------------------
# region FUNCTIONS

# endregion (functions)
# ---------------------------------------------------------
# region CLASSES


class Operand:

    def __init__(self, value: Any, label: Optional[str] = None):
        self.value: Any = value
        self.label: Optional[str] = label


class Constant(Operand):

    def __init__(self, value: int):
        super(Constant, self).__init__(value)

    def __str__(self):
        return str(self.value)


class CodeAddress(Operand):

    def __init__(self, address: Optional['Instruction'], label: str):
        super(CodeAddress, self).__init__(address, label)

    def __str__(self):
        return self.label


class HeapAddress(Operand):

    def __init__(self, value: int):
        super(HeapAddress, self).__init__(value)

    def __str__(self):
        return f"[{self.value}]"


class StackOffset(Operand):

    def __init__(self, value: int):
        super(StackOffset, self).__init__(value)

    def __str__(self):
        return f"({self.value})"


class Instruction:

    LOOP_COUNTER = 5000

    def __init__(self, address: int, opcode: Opcode, *args: int, **kwargs: Any):
        self.opcode: Opcode = opcode
        self.args: Any = args
        self.translate: List[str] = kwargs.get('translate', [])
        self.label: Optional[str] = kwargs.get('label')
        self.address: int = address
        self.prev: Optional[Instruction] = None
        self.next: Optional[Instruction] = None

    @property
    def size(self) -> int:
        return len(self.args) + 1

    @property
    def values(self) -> Tuple[int, ...]:
        return self.opcode.value, *self.args

    @property
    def first(self) -> 'Instruction':
        i = self
        counter = self.LOOP_COUNTER
        while i.prev and counter > 0:
            i = i.prev
            counter -= 1
            assert counter >= 0, 'Infinite loop detected or Instruction.LOOP_COUNTER too small'
        return i

    @property
    def last(self) -> 'Instruction':
        i = self
        counter = self.LOOP_COUNTER
        while i.next and counter > 0:
            i = i.next
            counter -= 1
            assert counter >= 0, 'Infinite loop detected or Instruction.LOOP_COUNTER too small'
        return i

    def __repr__(self):
        label = f'{self.label}:\n' if self.label else ''
        address = "{0:04X}".format(self.address)
        arg_vals = ""
        if self.args:
            arg_vals = "".join(tuple("{0:8}".format(arg) for arg in self.args))
        arg_vals = "{0:16}".format(arg_vals)
        arg_labels = ''
        if self.translate:
            arg_labels = " ".join(self.translate)

        result = f"{label}\t{address} {self.opcode.value} {arg_vals} :: {self.opcode.name} {arg_labels}"
        return result

    def find_prev_label(self) -> Optional['Instruction']:
        pass

    def insert_before(self, instruction: 'Instruction'):
        if self.prev:
            instruction.prev = self.prev
            self.prev.next = instruction

        self.prev = instruction
        instruction.next = self

    def insert_after(self, instruction: 'Instruction'):
        if self.next:
            instruction.next = self.next
            self.next.prev = instruction

        self.next = instruction
        instruction.prev = self

    def emit_first(self, opcode: Opcode, *args: int, **kwargs: Any) -> 'Instruction':
        i = Instruction(0, opcode, *args, **kwargs)
        first = self.first
        first.insert_before(i)
        return i

    def emit_last(self, opcode: Opcode, *args: int, **kwargs: Any) -> 'Instruction':
        i = Instruction(0, opcode, *args, **kwargs)
        last = self.last
        last.insert_after(i)
        return i

    def emit_before(self, opcode: Opcode, *args: int, **kwargs: Any) -> 'Instruction':
        i = Instruction(0, opcode, *args, **kwargs)
        self.insert_before(i)
        return i

    def emit_after(self, opcode: Opcode, *args: int, **kwargs: Any) -> 'Instruction':
        i = Instruction(0, opcode, *args, **kwargs)
        self.insert_after(i)
        return i

    def move_before(self):
        if self.prev:
            a = self.prev
            b = self
            a_prev = a.prev
            b_next = b.next

            a.prev = b
            a.next = b_next
            b.prev = a_prev
            b.next = a

    def move_after(self):
        if self.next:
            a = self
            b = self.next
            a_prev = a.prev
            b_next = b.next

            a.prev = b
            a.next = b_next
            b.prev = a_prev
            b.next = a


class Block:

    def __init__(self, parent: 'Block'):
        self.parent: Optional['Block'] = parent
        self.code: List[Instruction] = []

    @property
    def values(self) -> Tuple[int, ...]:
        values = ()
        for i in self.code:
            values += i.values
        return values

    @property
    def end_address(self):
        if len(self.code):
            return self.code[-1].next
        elif self.parent:
            return self.parent.end_address
        else:
            return 0

    @property
    def start_address(self):
        if len(self.code):
            return self.code[0].address
        return 0

    @property
    def count(self):
        return len(self.code)

    @property
    def size(self):
        n = 0
        for i in self.code:
            n += i.size
        return n

    def find(self, opcode: Opcode) -> Optional[Instruction]:
        for i in self.code:
            if i.opcode is opcode:
                return i
        return None

    def emit(self, opcode: Opcode, *args: int, **kwargs: Any) -> Instruction:
        i = Instruction(self.end_address, opcode, *args, **kwargs)
        self.code.append(i)
        return i


class Compiler:

    def __init__(self, fname: str):
        self.fname: str = fname
        self.ir: List[str] = [NOP.name, NOP.name]
        self.scope: Optional[ScopeNode] = None
        self.bin: List[int] = [NOP, NOP]
        self.rod: List[int] = []
        self.labels: Dict[str: int] = {}
        self.label_counter: int = -1
        self.instructions = Instruction(0,JMP, 0, translate=['principal'])

    def get_label(self, prefix: str) -> str:
        self.label_counter += 1
        return f"{prefix}{bin(self.label_counter)[2:]}"

    def emit_label(self, label, offset: int = 0):
        self.ir.append(f"${label}:")
        self.labels[label] = len(self.bin) + offset

    def emit(self, op: Opcode, *args: str, **kwargs):
        translate = kwargs.get('translate')
        tabs = '\t' if len(args) == 2 else '\t\t'
        tab = '\t'

        if translate:
            self.ir.append(f'{op.value}\t{tab.join(args)}{tabs}:: {op.name} {" ".join(translate)}')
        else:
            self.ir.append(f'{op.value}\t{tab.join(args)}{tabs}:: {op.name} {" ".join(args)}')

        self.bin.append(op.value)
        for arg in args:
            self.bin.append(int(arg))

    @contextmanager
    def emit_jump(self, op: Opcode, label: str):
        ir_idx = len(self.ir)
        bin_idx = len(self.bin)
        self.ir.append('')
        self.bin.extend([op.value, 0])

        yield

        jump_label = self.get_label(label)
        tabs = '\t\t'
        addr = len(self.bin)

        self.emit_label(jump_label, -2)
        self.ir[ir_idx] = f'{op.value}\t{addr}{tabs}:: {op.name} {jump_label}'
        self.bin[bin_idx: bin_idx + 1] = [op.value, addr]

    def emit_entrypoint(self, name: str):
        self.bin[0] = JMP.value
        self.bin[1] = self.labels[name]
        del self.ir[:2]
        self.ir.insert(0, f"{JMP.value}\t{self.labels[name]}\t:: {JMP.name} {name}")

    def compile(self):
        parser = Parser(self.fname)
        parser.parse()
        self.scope = parser.scope
        self.compile_ast()

    def compile_ast(self):
        print('compiling ast...')
        self.compile_scope(self.scope)
        self.emit(HALT)
        print('compiled\n\n')
        self.print_ir()
        print(self.bin)

    def print_ir(self):
        addr = 0
        for i in self.ir:
            if i.startswith('$'):
                print(i[1:])
            else:
                print(f"\t{addr}\t{i}")
                addr += 1

    def compile_scope(self, scope: ScopeNode):
        for name in scope.locals:
            if scope.locals[name].kind is NK_VAR_DECL:
                self.compile_decl_var(scope.locals[name])
        self.compile_block(scope)

    def compile_block(self, block_node: ScopeNode):
        for node in block_node.code:  # type: Union[StmtNode, DeclNode, ExprNode]
            if node.kind in NK_STATEMENTS:
                self.compile_stmt(node)
            elif node.kind is NK_FUNCTION_DECL:
                self.compile_decl_function(node)
            elif node.kind is NK_FCALL_EXPR:
                self.compile_expr_fcall(node)

    def compile_decl(self, decl_node: DeclNode):
        if decl_node.kind is NK_VAR_DECL:
            self.compile_decl_var(decl_node)

    def compile_decl_function(self, decl_node: DeclNode):
        self.emit_label(decl_node.name)
        if decl_node.is_main:
            self.emit_entrypoint(decl_node.name)
        self.compile_scope(cast(ScopeNode, decl_node.definition))

    def compile_decl_var(self, decl_node: DeclNode):
        self.compile_expr(cast(ExprNode, decl_node.initializer))

    def compile_stmt(self, stmt_node: StmtNode):
        if stmt_node.kind is NK_PRINT_STMT:
            self.compile_stmt_print(stmt_node)
        elif stmt_node.kind is NK_ASSIGN_STMT:
            self.compile_stmt_assign(stmt_node)
        elif stmt_node.kind is NK_RETURN_STMT:
            self.compile_stmt_return(stmt_node)
        elif stmt_node.kind is NK_IF_THEN_STMT:
            self.compile_stmt_ifthen(stmt_node)
        elif stmt_node.kind is NK_IF_ELSE_STMT:
            self.compile_stmt_ifelse(stmt_node)

    def compile_stmt_print(self, stmt_node: StmtNode):
        self.compile_expr(stmt_node['expr'])
        self.emit(IRQ, '1')

    def compile_stmt_return(self, stmt_node: StmtNode):
        self.compile_expr(stmt_node.nodes['expr'])
        self.emit(RET)

    def compile_stmt_assign(self, stmt_node: StmtNode):
        self.compile_expr(stmt_node.nodes['expr'])
        self.emit(SET, str(stmt_node['decl'].offset))

    def compile_stmt_ifthen(self, stmt_node: StmtNode):
        self.compile_expr(stmt_node['expr'])
        with self.emit_jump(JZ, 'end'):
            self.compile_scope(stmt_node['thenscope'])

    def compile_stmt_ifelse(self, stmt_node: StmtNode):
        self.compile_expr(stmt_node['expr'])
        with self.emit_jump(JZ, 'else'):
            self.compile_scope(stmt_node['thenscope'])
        with self.emit_jump(JMP, 'end'):
            self.compile_scope(stmt_node['elsescope'])

    def compile_expr(self, expr_node: ExprNode):
        if expr_node.kind is NK_BINARY_EXPR:
            self.compile_expr_binary(expr_node)
        elif expr_node.kind is NK_UNARY_EXPR:
            self.compile_expr_unary(expr_node)
        elif expr_node.kind is NK_INT32_EXPR:
            self.compile_expr_i32(expr_node)
        elif expr_node.kind is NK_INT64_EXPR:
            self.compile_expr_i64(expr_node)
        elif expr_node.kind is NK_VAR_EXPR:
            self.compile_expr_var(expr_node)
        elif expr_node.kind is NK_PARAM_EXPR:
            self.compile_expr_param(expr_node)
        elif expr_node.kind is NK_FCALL_EXPR:
            self.compile_expr_fcall(expr_node)
        elif expr_node.kind is NK_COMPARISON_EXPR:
            self.compile_expr_comparison(expr_node)

    def compile_expr_binary(self, expr_node: ExprNode):
        self.compile_expr(expr_node['right'])
        self.compile_expr(expr_node['left'])
        opcode = {
            '+': ADD, '-': SUB,
            '*': MUL, '/': DIV, '%': MOD
        }.get(expr_node.op, '?')
        self.emit(opcode)

    def compile_expr_unary(self, expr_node: ExprNode):
        self.compile_expr(expr_node['operand'])
        opcode = {
            '+': POS, '-': NEG,
        }.get(expr_node.op, '?')
        self.emit(opcode)

    def compile_expr_i32(self, expr_node: ExprNode):
        self.emit(CONST_D, expr_node['value'])

    def compile_expr_i64(self, expr_node: ExprNode):
        self.emit(CONST_FD, expr_node['value'])

    def compile_expr_var(self, expr_node: ExprNode):
        self.emit(GET, str(expr_node['decl'].offset))

    def compile_expr_param(self, expr_node: ExprNode):
        self.emit(GET, str(expr_node['decl'].offset))

    def compile_expr_fcall(self, expr_node: ExprNode):
        name = expr_node['decl'].name
        pointer = self.labels[name]
        argc = str(expr_node['argc'])
        for arg_node in expr_node['args']:   # type: ExprNode
            self.compile_expr(arg_node)
        self.emit(CALL, str(pointer), argc,
                  translate=[name, argc])

    def compile_expr_comparison(self, expr_node: ExprNode):
        self.compile_expr(expr_node['right'])
        self.compile_expr(expr_node['left'])
        opcode = {
            '<': LT, '<=': LTE,
            '>': GT, '>=': GTE,
            '==': EQ, '!=': NE,
        }.get(expr_node.op, '?')
        self.emit(opcode)

# endregion (classes)
# ---------------------------------------------------------
