from typing import List
from timeit import default_timer as timer
from brah.constants.mnemonics import *
from brah.d_compiler import Compiler

__all__ = [
    'Interpreter',
]

# ---------------------------------------------------------
# region CONSTANTS & ENUMS

PRT_CODE = 0
PRT_BYTECODE = 1

# endregion (constants)
# ---------------------------------------------------------
# region FUNCTIONS

# endregion (functions)
# ---------------------------------------------------------
# region CLASSES


class Interpreter:

    def __init__(self, program: str, as_bytecode: bool = False):
        self.program: List[int] = []

        if not as_bytecode:
            self.compile(program)
        else:
            # with open(program, 'rb') as bytecode:
            #     self.program = bytecode.read()
            self.program = [HALT]

    def compile(self, fname: str) -> None:
        compiler = Compiler(fname)

        compiler.compile()

        self.program = compiler.bin

    def execute(self) -> float:
        # return 0.0
        prog_start = timer()

        code: List[int] = self.program.copy()
        heap: List[int] = [0 for _ in range(256)]
        stack: List[int] = [0 for _ in range(256)]

        ip: int = 0
        sp: int = -1
        fp: int = 0

        zf: bool = False
        sf: bool = False

        op_name = ''

        while True:
            opcode = code[ip]
            op_name = OPCODES[opcode] if isinstance(opcode, int) else ''
            ip += 1
            if opcode == HALT:
                break

            elif opcode == NOP:
                pass

            elif opcode == IRQ:
                syscall = code[ip]
                ip += 1
                if syscall == 1:
                    print(stack[sp])
                    sp -= 1
            elif opcode in CONST:
                # get a immediate operand and increase IP
                sp += 1
                stack[sp] = code[ip]
                ip += 1

            elif opcode == GET:
                # push auto var to the top
                sp += 1
                stack[sp] = stack[fp + code[ip]]
                ip += 1

            elif opcode == SET:
                # pop top to auto var
                stack[fp + code[ip]] = stack[sp]
                sp -= 1
                ip += 1
            elif opcode == NEG:
                # pop top to auto var
                stack[sp] = -stack[sp]

            elif opcode == INC:
                stack[fp + code[ip]] += 1
                ip += 1

            elif opcode == DEC:
                stack[fp + code[ip]] -= 1
                ip += 1

            elif opcode in (ADD, SUB, MUL, DIV, MOD):
                l_op = stack[sp]
                sp -= 1
                r_op = stack[sp]
                sp -= 1
                sp += 1
                if opcode == ADD:
                    stack[sp] = l_op + r_op
                elif opcode == SUB:
                    stack[sp] = l_op - r_op
                elif opcode == MUL:
                    stack[sp] = l_op * r_op
                elif opcode == DIV:
                    stack[sp] = l_op // r_op
                else:  # opcode == MOD:
                    stack[sp] = l_op % r_op

            elif opcode in (EQ, NE, LT, LTE, GT, GTE):
                l_op = stack[sp]
                sp -= 1
                r_op = stack[sp]
                # skip last pointer decrement, due to increment right after
                if opcode == EQ:
                    stack[sp] = int(l_op == r_op)
                elif opcode == NE:
                    stack[sp] = int(l_op != r_op)
                elif opcode == LT:
                    stack[sp] = int(l_op < r_op)
                elif opcode == LTE:
                    stack[sp] = int(l_op <= r_op)
                elif opcode == GT:
                    stack[sp] = int(l_op > r_op)
                else:  # opcode == GTE:
                    stack[sp] = int(l_op >= r_op)

            elif opcode in (JZ, JNZ):
                cmp = stack[sp]
                sp -= 1
                addr = code[ip]
                ip += 1
                if opcode == JZ:
                    if not cmp:
                        ip = addr
                else:  # opcode == JNZ:
                    if cmp:
                        ip = addr

            elif opcode == CALL:
                # save the function pointer
                addr = code[ip]
                ip += 1
                # push number of args
                sp += 1
                frame = sp
                stack[sp] = code[ip]
                ip += 1
                # push the return address
                sp += 1
                stack[sp] = ip
                # push the frame pointer
                sp += 1
                stack[sp] = fp
                fp = frame
                # jump to function
                ip = addr

            elif opcode == RET:
                # save the return value
                ret = stack[sp]
                # clear the frame stack
                sp = fp + 2
                # pop the frame pointer
                fp = stack[sp]
                sp -= 1
                # pop the return address
                ip = stack[sp]
                sp -= 1
                # pop the number of args
                sp -= stack[sp]
                sp -= 1
                # push back the return value
                sp += 1
                stack[sp] = ret

            elif opcode == JMP:
                ip = code[ip]
                # sp -= 1

        time = timer() - prog_start
        return time
# endregion (classes)
# ---------------------------------------------------------
