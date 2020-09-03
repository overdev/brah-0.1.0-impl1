from typing import Optional, List, Dict, Union, cast, Any, Tuple
from brah.constants.tokens import SW_MAINFUNCTION, KW_BREAK, KW_CONTINUE, OP_INC
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
        self._value: Any = value
        self.label: Optional[str] = label

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val


class Constant(Operand):

    def __init__(self, value: Union[int, float]):
        super(Constant, self).__init__(value)

    def __str__(self):
        return str(self.value)


class CodeAddress(Operand):

    def __init__(self, address: 'Instruction', label: str, forward: bool):
        super(CodeAddress, self).__init__(address, label)
        address.owner = self
        self.forward: bool = forward

    def __str__(self):
        return self.label

    @property
    def instr(self) -> 'Instruction':
        return self._value

    @property
    def value(self):
        return self._value.address

    @value.setter
    def value(self, val):
        self._value = val

    def replace(self, with_instr: 'Instruction'):
        self._value.owner = None
        self._value = with_instr
        self._value.owner = self


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

    @classmethod
    def print_ir(cls, base_addr: int, instr: 'Instruction', out: Any = None):
        cls.bake_address(base_addr, instr)
        i: Instruction = instr
        inter_repr: str = 'ADDR     OPC  ARG1 ARG2 ARG3 :: OPCODE ARGS...'
        while i:
            inter_repr = f"{inter_repr}\n{i.ir()}"
            i = i.next
        print(inter_repr, file=out)

    @classmethod
    def bake_address(cls, base_addr: int, instr: 'Instruction', limit: int = 1000) -> int:
        i = instr
        counter = 0
        while i:
            i.address = base_addr
            base_addr += i.size
            i = i.next
            counter += 1
            if counter > limit:
                return counter
        return counter

    @classmethod
    def bake_code(cls, base_addr: int, instr: 'Instruction') -> List[int]:
        cls.bake_address(base_addr, instr)
        code = []
        index = 0
        i = instr
        while i:
            code.extend(i.values)
            i.index = index
            index += 1
            i = i.next
        return code

    @classmethod
    def invalid(cls, label: str) -> 'Instruction':
        return cls(0, NOP, Constant(0), Constant(0), label=label)

    def __init__(self, address: int, opcode: Opcode, *args: Operand, label: Optional[str]):
        self.opcode: Opcode = opcode
        self.args: Tuple[Operand, ...] = args
        self.label: Optional[str] = label
        self.address: int = address
        self.index: int = 0
        self.prev: Optional[Instruction] = None
        self.next: Optional[Instruction] = None
        self.owner: Optional[CodeAddress] = None

    def __repr__(self):
        return f"({self.opcode.name}, {self.size}, {self.label if self.label else '-'}, {bool(self.owner)}, [{self.address}])"

    @property
    def size(self) -> int:
        return len(self.args) + 1

    @property
    def values(self) -> Tuple[int, ...]:
        return self.opcode.value, *(o.value for o in self.args)

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
    def is_invalid(self) -> bool:
        return self.opcode is NOP and self.size == 3

    @property
    def last(self) -> 'Instruction':
        i = self
        counter = self.LOOP_COUNTER
        while i.next and counter > 0:
            i = i.next
            counter -= 1
            assert counter >= 0, 'Infinite loop detected or Instruction.LOOP_COUNTER too small'
        return i

    def ir(self) -> str:
        label = f'{self.label}:\n' if self.label else ''
        address = "{0:04}".format(self.address)
        opcode = "{0:>4}".format(self.opcode.value)
        # arg_vals = " "
        # for a in self.args:
        #     if isinstance(a, CodeAddress):
        #         print(a.value)
        arg_vals = " ".join(tuple("{0:>4}".format(arg.value) for arg in self.args))
        arg_vals = "{0:14}".format(arg_vals)
        arg_labels = " ".join(tuple("{0:>4}".format(str(arg)) for arg in self.args))

        result = f"{label}\t{address} {opcode} {arg_vals} :: {self.opcode.name} {arg_labels}"
        return result

    def find_label(self, label: str, forward: bool) -> Optional['Instruction']:
        i = self
        while i:
            if i.label == label:
                return i
            i = i.next if forward else i.prev

    def insert_before(self, instruction: 'Instruction') -> 'Instruction':
        if self.is_invalid:
            instruction.label = self.label
            instruction.prev = self.prev
            instruction.next = self.next
            if self.prev:
                self.prev.next = instruction
            if self.next:
                self.next.prev = instruction
            self.prev = self.next = None
            if self.owner:
                self.owner.replace(instruction)
        else:
            if self.prev:
                instruction.prev = self.prev
                self.prev.next = instruction

            self.prev = instruction
            instruction.next = self
        return instruction

    def insert_after(self, instruction: 'Instruction') -> 'Instruction':
        if self.is_invalid:
            instruction.label = self.label
            instruction.prev = self.prev
            instruction.next = self.next
            if self.prev:
                self.prev.next = instruction
            if self.next:
                self.next.prev = instruction
            self.prev = self.next = None
            if self.owner:
                self.owner.replace(instruction)
        else:
            if self.next:
                instruction.next = self.next
                self.next.prev = instruction

            self.next = instruction
            instruction.prev = self
        return instruction

    def emit_first(self, opcode: Opcode, *args: Operand, label: Optional[str] = None) -> 'Instruction':
        first = self.first
        return first.insert_before(Instruction(0, opcode, *args, label=label))

    def emit_last(self, opcode: Opcode, *args: Operand, label: Optional[str] = None) -> 'Instruction':
        last = self.last
        return last.insert_after(Instruction(0, opcode, *args, label=label))

    def emit_before(self, opcode: Opcode, *args: Operand, label: Optional[str] = None) -> 'Instruction':
        return self.insert_before(Instruction(0, opcode, *args, label=label))

    def emit_after(self, opcode: Opcode, *args: Operand, label: Optional[str] = None) -> 'Instruction':
        return self.insert_after(Instruction(0, opcode, *args, label=label))

    def emit_before_invalid(self, label: str) -> 'Instruction':
        return self.insert_before(Instruction.invalid(label))

    def emit_after_invalid(self, label: str) -> 'Instruction':
        return self.insert_after(Instruction.invalid(label))

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


class Compiler:

    def __init__(self, fname: str):
        self.fname: str = fname
        # self.ir: List[str] = [NOP.name, NOP.name]
        self.scope: Optional[ScopeNode] = None
        self.bin: List[int] = [NOP, NOP]
        self.rod: List[int] = []
        # self.labels: Dict[str: int] = {}
        # self.label_counter: int = -1
        self.func_decls: Dict[str, CodeAddress] = {}
        self.jump_stmts: Dict[str, CodeAddress] = {}
        code_addr = self.get_call_instr(SW_MAINFUNCTION)
        self.instructions = Instruction(0, CALL, code_addr, Constant(0), label=None)

    #
    # def get_label(self, prefix: str) -> str:
    #     self.label_counter += 1
    #     return f"{prefix}{bin(self.label_counter)[2:]}"
    #
    # def emit_label(self, label, offset: int = 0):
    #     self.ir.append(f"${label}:")
    #     self.labels[label] = len(self.bin) + offset
    #
    # def emit(self, op: Opcode, *args: str, **kwargs):
    #     translate = kwargs.get('translate')
    #     tabs = '\t' if len(args) == 2 else '\t\t'
    #     tab = '\t'
    #
    #     if translate:
    #         self.ir.append(f'{op.value}\t{tab.join(args)}{tabs}:: {op.name} {" ".join(translate)}')
    #     else:
    #         self.ir.append(f'{op.value}\t{tab.join(args)}{tabs}:: {op.name} {" ".join(args)}')
    #
    #     self.bin.append(op.value)
    #     for arg in args:
    #         self.bin.append(int(arg))
    #
    # @contextmanager
    # def emit_jump(self, op: Opcode, label: str):
    #     ir_idx = len(self.ir)
    #     bin_idx = len(self.bin)
    #     self.ir.append('')
    #     self.bin.extend([op.value, 0])
    #
    #     yield
    #
    #     jump_label = self.get_label(label)
    #     tabs = '\t\t'
    #     addr = len(self.bin)
    #
    #     self.emit_label(jump_label, -2)
    #     self.ir[ir_idx] = f'{op.value}\t{addr}{tabs}:: {op.name} {jump_label}'
    #     self.bin[bin_idx: bin_idx + 1] = [op.value, addr]
    #
    # def emit_entrypoint(self, name: str):
    #     self.bin[0] = JMP.value
    #     self.bin[1] = self.labels[name]
    #     del self.ir[:2]
    #     self.ir.insert(0, f"{JMP.value}\t{self.labels[name]}\t:: {JMP.name} {name}")
    #
    # def print_ir(self):
    #     addr = 0
    #     for i in self.ir:
    #         if i.startswith('$'):
    #             print(i[1:])
    #         else:
    #             print(f"\t{addr}\t{i}")
    #             addr += 1

    def get_call_instr(self, name: str) -> CodeAddress:
        if name not in self.func_decls:
            self.func_decls[name] = CodeAddress(Instruction.invalid(name), name, True)
        return self.func_decls[name]

    def set_call_instr(self, name: str, instr: Instruction):
        if name not in self.func_decls:
            self.func_decls[name] = CodeAddress(instr, name, True)
        else:
            self.func_decls[name].replace(instr)

    def clear_jumps(self):
        self.jump_stmts.clear()

    def get_jump(self, name: str) -> CodeAddress:
        if name not in self.jump_stmts:
            self.jump_stmts[name] = CodeAddress(Instruction.invalid(name), name, True)
        return self.jump_stmts[name]

    def set_jump(self, name: str, instr: Instruction):
        if name not in self.jump_stmts:
            self.jump_stmts[name] = CodeAddress(instr, name, True)
        else:
            self.jump_stmts[name].replace(instr)

    def compile(self):
        parser = Parser(self.fname)
        parser.parse(AssemblyNode())
        self.scope = parser.scope
        self.compile_ast()

    def compile_ast(self):
        print('compiling ast...\n\n')
        instr = self.instructions.emit_after(HALT)
        self.compile_scope(self.scope, None, instr)

        print('generating bytecode..\n\n')
        self.bin = Instruction.bake_code(0, self.instructions)

        print(self.bin)
        print(f"{len(self.bin)} elements in {self.instructions.last.index} instructions")
        print('\nprinting IR...\n')

        fname = 'C:\\Jorge\\Github\\Python\\StackMachines\\brah-0.1.0-impl1\\examples\\arith.txt'
        with open(fname, 'w', encoding='utf8') as file:
            Instruction.print_ir(0, self.instructions, file)
        print('finished.')

    def compile_scope(self, scope: ScopeNode, label: Optional[str], instr: Instruction) -> Instruction:
        # for name in scope.locals:
        #     if scope.locals[name].kind is NK_VAR_DECL:
        #         instr = self.compile_decl_var(scope.locals[name], instr)
        for decl in scope.initializers:  # type: DeclNode
            if decl.initializer:
                instr = self.compile_expr(decl.initializer, instr)
            # else:
            #     instr = instr.emit_after(CONST_D, Constant(0))
        return self.compile_block(scope, instr)

    def compile_block(self, block_node: ScopeNode, instr: Instruction) -> Instruction:
        for node in block_node.code:  # type: Union[StmtNode, DeclNode, ExprNode]
            if node.kind in NK_STATEMENTS:
                instr = self.compile_stmt(node, instr)
            elif node.kind is NK_FUNCTION_DECL:
                instr = self.compile_decl_function(node, instr)
            elif node.kind is NK_FCALL_EXPR:
                instr = self.compile_expr_fcall(node, instr)
        return instr

    def compile_decl(self, decl_node: DeclNode, instr: Instruction) -> Instruction:
        if decl_node.kind is NK_VAR_DECL:
            return self.compile_decl_var(decl_node, instr)

    def compile_decl_function(self, decl_node: DeclNode, instr: Instruction) -> Instruction:
        # self.emit_label(decl_node.name)
        # if decl_node.is_main:
        #     self.emit_entrypoint(decl_node.name)
        # instr = instr.emit_after_invalid(decl_node.name)
        code_addr: CodeAddress = self.get_call_instr(decl_node.name)
        instr = instr.insert_after(code_addr.instr)
        instr = self.compile_scope(cast(ScopeNode, decl_node.definition), decl_node.name, instr)
        return instr.emit_after(RET)

    def compile_decl_var(self, decl_node: DeclNode, instr: Instruction) -> Instruction:
        return self.compile_expr(cast(ExprNode, decl_node.initializer), instr)
        # return instr.emit_after(SET, StackOffset(decl_node.offset))

    def compile_stmt(self, stmt_node: StmtNode, instr: Instruction) -> Instruction:
        if stmt_node.kind is NK_PRINT_STMT:
            instr = self.compile_stmt_print(stmt_node, instr)
        elif stmt_node.kind is NK_ASSIGN_STMT:
            instr = self.compile_stmt_assign(stmt_node, instr)
        elif stmt_node.kind in (NK_INC_STMT, NK_DEC_STMT):
            instr = self.compile_stmt_incr(stmt_node, instr)
        elif stmt_node.kind is NK_RETURN_STMT:
            instr = self.compile_stmt_return(stmt_node, instr)
        elif stmt_node.kind is NK_CONTINUE_STMT:
            instr = self.compile_stmt_continue(stmt_node, instr)
        elif stmt_node.kind is NK_BREAK_STMT:
            instr = self.compile_stmt_break(stmt_node, instr)
        elif stmt_node.kind is NK_IF_THEN_STMT:
            instr = self.compile_stmt_ifthen(stmt_node, instr)
        elif stmt_node.kind is NK_IF_ELSE_STMT:
            instr = self.compile_stmt_ifelse(stmt_node, instr)
        elif stmt_node.kind is NK_WHILE_STMT:
            instr = self.compile_stmt_while(stmt_node, instr)
        elif stmt_node.kind is NK_DO_WHILE_STMT:
            instr = self.compile_stmt_do_while(stmt_node, instr)
        elif stmt_node.kind is NK_DO_UNTIL_STMT:
            instr = self.compile_stmt_do_until(stmt_node, instr)
        elif stmt_node.kind is NK_REPEAT_FINITE_STMT:
            instr = self.compile_stmt_repeat(stmt_node, instr)
        return instr

    def compile_stmt_print(self, stmt_node: StmtNode, instr: Instruction) -> Instruction:
        instr = self.compile_expr(stmt_node['expr'], instr)
        return instr.emit_after(IRQ, Constant(1))

    def compile_stmt_return(self, stmt_node: StmtNode, instr: Instruction) -> Instruction:
        instr = self.compile_expr(stmt_node.nodes['expr'], instr)
        return instr.emit_after(RET)

    def compile_stmt_assign(self, stmt_node: StmtNode, instr: Instruction) -> Instruction:
        instr = self.compile_expr(stmt_node.nodes['expr'], instr)
        return instr.emit_after(SET, StackOffset(stmt_node['decl'].offset))

    def compile_stmt_incr(self, stmt_node: StmtNode, instr: Instruction) -> Instruction:
        instr = self.compile_expr(stmt_node.nodes['decl'], instr)
        op = INC if stmt_node.kind == NK_INC_STMT else DEC
        return instr.emit_after(op, StackOffset(stmt_node['decl'].offset))

    def compile_stmt_ifthen(self, stmt_node: StmtNode, instr: Instruction) -> Instruction:
        instr = self.compile_expr(stmt_node['expr'], instr)
        label = 'end'
        i = instr.emit_after(JZ, CodeAddress(Instruction.invalid(label), label, True))
        jmp = i
        i = self.compile_scope(stmt_node['thenscope'], None, i)
        end = i.emit_after(NOP, label=label)
        jmp.args[0].value = end
        return end

    def compile_stmt_ifelse(self, stmt_node: StmtNode, instr: Instruction) -> Instruction:
        instr = self.compile_expr(stmt_node['expr'], instr)
        end_label = 'end'
        else_label = 'else'
        then_tgt = Instruction.invalid(else_label)
        end_tgt = Instruction.invalid(end_label)
        then_jmp = instr.emit_after(JZ, CodeAddress(then_tgt, else_label, True))
        instr = self.compile_scope(stmt_node['thenscope'], None, then_jmp)
        end_jmp = instr.emit_after(JMP, CodeAddress(end_tgt, end_label, True))
        instr = end_jmp.insert_after(then_tgt)
        instr = self.compile_scope(stmt_node['elsescope'], None, instr)
        return instr.insert_after(end_tgt)

    def compile_stmt_while(self, stmt_node: StmtNode, instr: Instruction) -> Instruction:
        whilescope = stmt_node['whilescope']
        start_label = stmt_node.label
        end_label = f"{stmt_node.label}End"
        start: CodeAddress = self.get_jump(start_label)
        end: CodeAddress = self.get_jump(end_label)

        whilescope.set_label(KW_BREAK, end_label)
        whilescope.set_label(KW_CONTINUE, end_label)

        instr = instr.insert_after(start.instr)
        instr = self.compile_expr(stmt_node['expr'], instr)
        instr = instr.emit_after(JZ, end)
        instr = self.compile_scope(whilescope, None, instr)
        instr = instr.emit_after(JMP, start)
        return instr.insert_after(end.instr)

    def compile_stmt_do_while(self, stmt_node: StmtNode, instr: Instruction) -> Instruction:
        start: CodeAddress = self.get_jump(stmt_node.label)

        instr = instr.insert_after(start.instr)
        instr = self.compile_scope(stmt_node['doscope'], None, instr)
        instr = self.compile_expr(stmt_node['expr'], instr)
        return instr.emit_after(JNZ, start)

    def compile_stmt_do_until(self, stmt_node: StmtNode, instr: Instruction) -> Instruction:
        start: CodeAddress = self.get_jump(stmt_node.label)

        instr = instr.insert_after(start.instr)
        instr = self.compile_scope(stmt_node['doscope'], None, instr)
        instr = self.compile_expr(stmt_node['expr'], instr)
        return instr.emit_after(JZ, start)

    def compile_stmt_repeat(self, stmt_node: StmtNode, instr: Instruction) -> Instruction:
        start: CodeAddress = self.get_jump(stmt_node.label)
        end: CodeAddress = self.get_jump(f"{stmt_node.label}End")

        scope: ScopeNode = stmt_node['repeatscope']
        has_counter = stmt_node.kind is NK_REPEAT_FINITE_STMT
        if has_counter:
            for stmt in scope.loopcounters:
                instr = self.compile_stmt_assign(stmt, instr)
            idx_node: DeclNode = scope.locals[scope.iteration['start']]
            max_node: DeclNode = scope.locals[scope.iteration['stop']]
            instr = instr.insert_after(start.instr)
            instr = instr.emit_after(GET, StackOffset(idx_node.offset))
            instr = instr.emit_after(GET, StackOffset(max_node.offset))
            instr = instr.emit_after(GT)
            instr = instr.emit_after(JZ, end)

        instr = self.compile_scope(scope, None, instr)
        if has_counter:
            idx_node: DeclNode = scope.locals[scope.iteration['start']]
            instr = instr.emit_after(INC, StackOffset(idx_node.offset))
        instr = instr.emit_after(JMP, start)
        return instr.insert_after(end.instr)

    def compile_stmt_break(self, stmt_node: StmtNode, instr: Instruction) -> Instruction:
        end: CodeAddress = self.get_jump(f"{stmt_node.label}End")
        return instr.emit_after(JMP, end)

    def compile_stmt_continue(self, stmt_node: StmtNode, instr: Instruction) -> Instruction:
        start: CodeAddress = self.get_jump(f"{stmt_node.label}")
        return instr.emit_after(JMP, start)

    def compile_expr(self, expr_node: ExprNode, instr: Instruction) -> Instruction:
        if expr_node.kind is NK_BINARY_EXPR:
            instr = self.compile_expr_binary(expr_node, instr)
        elif expr_node.kind is NK_UNARY_EXPR:
            instr = self.compile_expr_unary(expr_node, instr)
        elif expr_node.kind is NK_INT8_EXPR:
            instr = self.compile_expr_i8(expr_node, instr)
        elif expr_node.kind is NK_INT16_EXPR:
            instr = self.compile_expr_i16(expr_node, instr)
        elif expr_node.kind is NK_INT32_EXPR:
            instr = self.compile_expr_i32(expr_node, instr)
        elif expr_node.kind is NK_INT64_EXPR:
            instr = self.compile_expr_i64(expr_node, instr)
        elif expr_node.kind is NK_UINT8_EXPR:
            instr = self.compile_expr_u8(expr_node, instr)
        elif expr_node.kind is NK_UINT16_EXPR:
            instr = self.compile_expr_u16(expr_node, instr)
        elif expr_node.kind is NK_UINT32_EXPR:
            instr = self.compile_expr_u32(expr_node, instr)
        elif expr_node.kind is NK_UINT64_EXPR:
            instr = self.compile_expr_u64(expr_node, instr)
        elif expr_node.kind is NK_FLOAT16_EXPR:
            instr = self.compile_expr_f16(expr_node, instr)
        elif expr_node.kind is NK_FLOAT32_EXPR:
            instr = self.compile_expr_f32(expr_node, instr)
        elif expr_node.kind is NK_FLOAT64_EXPR:
            instr = self.compile_expr_f64(expr_node, instr)
        elif expr_node.kind is NK_FLOAT80_EXPR:
            instr = self.compile_expr_f80(expr_node, instr)
        elif expr_node.kind is NK_VAR_EXPR:
            instr = self.compile_expr_var(expr_node, instr)
        elif expr_node.kind is NK_PARAM_EXPR:
            instr = self.compile_expr_param(expr_node, instr)
        elif expr_node.kind is NK_FCALL_EXPR:
            instr = self.compile_expr_fcall(expr_node, instr)
        elif expr_node.kind in (NK_INC_EXPR, NK_DEC_EXPR):
            instr = self.compile_expr_incr(expr_node, instr)
        elif expr_node.kind is NK_COMPARISON_EXPR:
            instr = self.compile_expr_comparison(expr_node, instr)
        elif expr_node.kind is NK_UNDEFINED_EXPR:
            instr = instr.emit_after(CONST_D, Constant(0))
        return instr

    def compile_expr_binary(self, expr_node: ExprNode, instr: Instruction) -> Instruction:
        instr = self.compile_expr(expr_node['right'], instr)
        instr = self.compile_expr(expr_node['left'], instr)
        opcode = {
            '+': ADD, '-': SUB,
            '*': MUL, '/': DIV, '%': MOD
        }.get(expr_node.op, '?')
        return instr.emit_after(opcode)
        # self.emit(opcode)

    def compile_expr_unary(self, expr_node: ExprNode, instr: Instruction) -> Instruction:
        instr = self.compile_expr(expr_node['operand'], instr)
        opcode = {
            '+': POS, '-': NEG,
        }.get(expr_node.op, '?')
        return instr.emit_after(opcode)

    def compile_expr_i8(self, expr_node: ExprNode, instr: Instruction) -> Instruction:
        return instr.emit_after(CONST_SB, Constant(int(expr_node.value)))

    def compile_expr_i16(self, expr_node: ExprNode, instr: Instruction) -> Instruction:
        return instr.emit_after(CONST_SW, Constant(int(expr_node.value)))

    def compile_expr_i32(self, expr_node: ExprNode, instr: Instruction) -> Instruction:
        return instr.emit_after(CONST_SD, Constant(int(expr_node.value)))

    def compile_expr_i64(self, expr_node: ExprNode, instr: Instruction) -> Instruction:
        return instr.emit_after(CONST_SQ, Constant(int(expr_node.value)))

    def compile_expr_u8(self, expr_node: ExprNode, instr: Instruction) -> Instruction:
        return instr.emit_after(CONST_B, Constant(int(expr_node.value)))

    def compile_expr_u16(self, expr_node: ExprNode, instr: Instruction) -> Instruction:
        return instr.emit_after(CONST_W, Constant(int(expr_node.value)))

    def compile_expr_u32(self, expr_node: ExprNode, instr: Instruction) -> Instruction:
        return instr.emit_after(CONST_D, Constant(int(expr_node.value)))

    def compile_expr_u64(self, expr_node: ExprNode, instr: Instruction) -> Instruction:
        return instr.emit_after(CONST_Q, Constant(int(expr_node.value)))

    def compile_expr_f16(self, expr_node: ExprNode, instr: Instruction) -> Instruction:
        return instr.emit_after(CONST_FH, Constant(float(expr_node.value)))

    def compile_expr_f32(self, expr_node: ExprNode, instr: Instruction) -> Instruction:
        return instr.emit_after(CONST_FS, Constant(float(expr_node.value)))

    def compile_expr_f64(self, expr_node: ExprNode, instr: Instruction) -> Instruction:
        return instr.emit_after(CONST_FD, Constant(float(expr_node.value)))

    def compile_expr_var(self, expr_node: ExprNode, instr: Instruction) -> Instruction:
        return instr.emit_after(GET, StackOffset(expr_node['decl'].offset))

    def compile_expr_param(self, expr_node: ExprNode, instr: Instruction) -> Instruction:
        return instr.emit_after(GET, StackOffset(expr_node['decl'].offset))

    def compile_expr_fcall(self, expr_node: ExprNode, instr: Instruction) -> Instruction:
        name = expr_node['decl'].name
        code_addr = self.get_call_instr(name)
        # pointer = self.labels[name]
        argc = cast(int, expr_node['argc'])
        for arg_node in expr_node['args']:  # type: ExprNode
            instr = self.compile_expr(arg_node, instr)
        return instr.emit_after(CALL, code_addr, Constant(argc))
        # self.emit(CALL, str(pointer), argc, translate=[name, argc])

    def compile_expr_incr(self, expr_node: ExprNode, instr: Instruction) -> Instruction:
        instr = self.compile_expr(expr_node['operand'], instr)
        op = INC if expr_node.kind == NK_INC_EXPR else DEC
        return instr.emit_after(op, Constant(0))

    def compile_expr_comparison(self, expr_node: ExprNode, instr: Instruction) -> Instruction:
        instr = self.compile_expr(expr_node['right'], instr)
        instr = self.compile_expr(expr_node['left'], instr)
        opcode = {
            '<': LT, '<=': LTE,
            '>': GT, '>=': GTE,
            '==': EQ, '!=': NE,
        }.get(expr_node.op, '?')
        return instr.emit_after(opcode)
        # self.emit(opcode)

# endregion (classes)
# ---------------------------------------------------------
