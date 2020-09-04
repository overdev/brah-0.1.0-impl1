import random
from typing import Optional, List, Dict, Tuple, Union
from brah.constants.tokens import *
from brah.constants.nodes import *
from brah.b_lexer import *
from brah.a_scanner import Source, SCN_DECDIGITS

__all__ = [
    'Parser',
    'ASTNode',
    'AssemblyNode',
    'DeclNode',
    'StmtNode',
    'ExprNode',
    'TypeNode',
    'ScopeNode',
]

# ---------------------------------------------------------
# region CONSTANTS & ENUMS

DECL_TO_EXPR = {
    NK_VAR_DECL: NK_VAR_EXPR,
    NK_PARAM_DECL: NK_PARAM_EXPR,
    NK_FUNCTION_DECL: NK_FUNCTION_EXPR
}

# endregion (constants)
# ---------------------------------------------------------
# region FUNCTIONS

# endregion (functions)
# ---------------------------------------------------------
# region CLASSES


class ASTNode:

    def __init__(self, kind: NodeKind, **kwargs):
        self.kind: NodeKind = kind
        self.pos: Tuple[int, int] = 0, 0
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def __repr__(self):
        return f"{self.__class__.__name__}({self.kind.name})"

    def __str__(self):
        return f"({self.kind.name}: {self.pos})"


class DeclNode(ASTNode):

    def __init__(self, kind: NodeKind, name: str, **kwargs):
        super(DeclNode, self).__init__(kind)
        self.name: str = name
        self.type: Optional[TypeNode] = kwargs.get('type')
        self.definition: Optional[ASTNode] = None
        self.initializer: Optional[ExprNode] = None
        self.scope: Optional[ScopeNode] = None
        self.reads: int = 0
        self.writes: int = 0
        self.offset: int = 0
        self.params: List[DeclNode] = []
        self.is_main: bool = False

    def read(self) -> int:
        self.reads += 1
        return self.reads

    def write(self) -> int:
        self.writes += 1
        return self.writes


class StmtNode(ASTNode):

    def __init__(self, kind: NodeKind, **kwargs):
        super(StmtNode, self).__init__(kind, **kwargs)
        self.nodes: Dict[str, ExprNode] = kwargs.copy()
        self.label: Optional[str] = kwargs.get('label')

    def __getitem__(self, key) -> Union['ExprNode', 'DeclNode', 'ScopeNode']:
        return self.nodes.__getitem__(key)

    def __setitem__(self, key, value: Union['ExprNode', 'DeclNode']):
        self.nodes.__setitem__(key, value)


class ExprNode(ASTNode):

    def __init__(self, kind: NodeKind, **kwargs):
        super(ExprNode, self).__init__(kind, **kwargs)
        self.nodes: Dict[str, Union[str, ExprNode]] = kwargs.copy()
        self.op: str = kwargs.get('op', '')
        self.value: str = kwargs.get('value', '0')
        self.type: Optional[TypeNode] = kwargs.get('type')
        self.constant: Optional[Union[int, float]] = kwargs.get('constant')

    def __getitem__(self, key) -> Union['ExprNode', 'DeclNode']:
        return self.nodes.__getitem__(key)

    def __setitem__(self, key, value: Union['ExprNode', 'DeclNode']):
        self.nodes.__setitem__(key, value)


class TypeNode(ASTNode):

    def __init__(self, kind: NodeKind, **kwargs):
        super(TypeNode, self).__init__(kind, **kwargs)
        # all
        self.name: str = kwargs.get('name')
        self.size: int = kwargs.get('size', 1)

        # primitives
        self.integer: bool = kwargs.get('integer', True)
        self.signed: bool = kwargs.get('signed', True)

        # pointers, arrays and classes
        self.base: Optional[TypeNode] = kwargs.get('base')

        # arrays
        self.length: ExprNode = kwargs.get('length', ExprNode(NK_INT32_EXPR, value='-1'))

        # signatures and functions
        self.params: List[TypeNode] = kwargs.get('params', [])
        self.result: TypeNode = kwargs.get('result')

        # interfaces, structures and classes
        self.fields: List[TypeNode] = kwargs.get('fields', [])
        self.methods: List[TypeNode] = kwargs.get('methods', [])
        self.operators: List[TypeNode] = kwargs.get('methods', [])

    def accepts(self, other: 'TypeNode') -> bool:
        if other.kind is self.kind:
            kind = self.kind
            if kind is NK_PRIMITIVE_TYPE:
                # if self.size < other.size:
                #     return False
                if self.signed != other.signed:
                    return False
            elif kind is NK_FUNCTION_TYPE:
                if len(self.params) != len(other.params):
                    return False
                if not self.result.accepts(other.result):
                    return False
                for i in range(len(self.params)):
                    if not self.params[i].accepts(other.params[i]):
                        return False
            return True
        else:
            return False


class AssemblyNode(ASTNode):
    types: Dict[str, TypeNode] = {
        TY_VOID: TypeNode(NK_PRIMITIVE_TYPE, name=TY_VOID, signed=False, size=1, integer=True),
        TY_BOOL: TypeNode(NK_PRIMITIVE_TYPE, name=TY_BOOL, signed=False, size=1, integer=True),
        TY_INT8: TypeNode(NK_PRIMITIVE_TYPE, name=TY_INT8, signed=True, size=1, integer=True),
        TY_INT16: TypeNode(NK_PRIMITIVE_TYPE, name=TY_INT16, signed=True, size=2, integer=True),
        TY_INT32: TypeNode(NK_PRIMITIVE_TYPE, name=TY_INT32, signed=True, size=4, integer=True),
        TY_INT64: TypeNode(NK_PRIMITIVE_TYPE, name=TY_INT64, signed=True, size=8, integer=True),
        TY_UINT8: TypeNode(NK_PRIMITIVE_TYPE, name=TY_UINT8, signed=False, size=1, integer=True),
        TY_UINT16: TypeNode(NK_PRIMITIVE_TYPE, name=TY_UINT16, signed=False, size=2, integer=True),
        TY_UINT32: TypeNode(NK_PRIMITIVE_TYPE, name=TY_UINT32, signed=False, size=4, integer=True),
        TY_UINT64: TypeNode(NK_PRIMITIVE_TYPE, name=TY_UINT64, signed=False, size=8, integer=True),
        TY_FLOAT16: TypeNode(NK_PRIMITIVE_TYPE, name=TY_FLOAT16, signed=True, size=2, integer=False),
        TY_FLOAT32: TypeNode(NK_PRIMITIVE_TYPE, name=TY_FLOAT32, signed=True, size=4, integer=False),
        TY_FLOAT64: TypeNode(NK_PRIMITIVE_TYPE, name=TY_FLOAT64, signed=True, size=8, integer=False),
        TY_FLOAT80: TypeNode(NK_PRIMITIVE_TYPE, name=TY_FLOAT80, signed=True, size=10, integer=False),
    }

    def __init__(self, **kwargs):
        # assert kind in (NK_MODULE,)
        super(AssemblyNode, self).__init__(NK_NONE)
        self.modules: List[ScopeNode] = []
        self.target_fname: str = kwargs.get('target', '../output/out.brbc')
        self.source_dir: str = kwargs.get('src', '../src')

    @staticmethod
    def find(name: str) -> Optional[TypeNode]:
        return AssemblyNode.types.get(name)


class ScopeNode(ASTNode):
    FRAME_OFFSET = 3

    def __init__(self, kind: NodeKind, parent: Optional['ScopeNode'] = None, **kwargs):
        super(ScopeNode, self).__init__(kind)
        self._assembly: Optional[AssemblyNode] = kwargs.get('assembly')
        self.parent: Optional[ScopeNode] = parent
        self.decl: Optional[DeclNode] = kwargs.get('decl')
        self.locals: Dict[str, DeclNode] = {}
        self.code: List[Union[StmtNode, DeclNode]] = []
        self.loopcounters: List[StmtNode] = []
        self.iteration: Dict[str, str] = {}
        self.offset: int = 0
        self.jump_labels: Dict[str, int] = {}
        self.default_labels: Dict[str, str] = {}
        self.initializers: List[DeclNode] = []
        self.types: Dict[str, TypeNode] = {}

    @property
    def assembly(self) -> Optional[AssemblyNode]:
        if self.kind is NK_MODULE_SCOPE:
            return self._assembly
        else:
            assert self.parent, "Assembly node is not defined."
            return self.parent.assembly

    @assembly.setter
    def assembly(self, value: AssemblyNode):
        self._assembly = value

    @property
    def in_function_scope(self) -> bool:
        return self.find_scope(NK_FUNCTION_SCOPE) is not None

    @property
    def in_method_scope(self) -> bool:
        return self.find_scope(NK_METHOD_SCOPE) is not None

    @property
    def in_class_scope(self) -> bool:
        return self.find_scope(NK_CLASS_SCOPE) is not None

    @property
    def base_scope(self) -> Optional['ScopeNode']:
        if self.kind in (NK_STATEMENT_SCOPE, NK_LOOP_SCOPE, NK_CASE_SCOPE):
            assert self.parent is not None, "Fatal Error: STATEMENT_SCOPE not parented by another scope"
            return self.parent.base_scope
        else:
            return self

    def set_iteration(self, **kwargs):
        self.iteration.update(kwargs)

    def auto_define_label(self, prefix: str) -> str:
        if prefix in self.jump_labels:
            self.jump_labels[prefix] += 1
            return f"{prefix}{self.jump_labels[prefix]}"
        else:
            self.jump_labels[prefix] = 0
            return prefix

    def define_label(self, label: str) -> bool:
        if label in self.jump_labels:
            return False
        self.jump_labels[label] = 0
        return True

    def has_label(self, label: str) -> bool:
        return label in self.jump_labels

    def set_label(self, stmt_kw: str, label: str):
        self.default_labels[stmt_kw] = label

    def get_label(self, stmt_kw: str) -> Optional[str]:
        if self.kind in (NK_CASE_SCOPE, NK_LOOP_SCOPE):
            if stmt_kw in self.default_labels:
                return self.default_labels[stmt_kw]
            else:
                return self.parent.get_label(stmt_kw)
        elif self.parent:
            return self.parent.get_label(stmt_kw)
        else:
            return None

    def find_scope(self, *kinds: NodeKind) -> Optional['ScopeNode']:
        if self.kind in kinds:
            return self
        elif self.parent:
            return self.parent.find_scope(*kinds)
        else:
            return None

    @staticmethod
    def gen_name(prefix: str) -> str:
        return f"__{prefix}_{hex(random.randint(100000, 999999))[2:]}"

    def declare(self, name: str, node: DeclNode, *expected_scopes: NodeKind) -> bool:
        if name in self.locals:
            return False

        self.locals[name] = node
        base = self.find_scope(*expected_scopes)
        scopes = ", nor outside of ".join(n.name for n in expected_scopes)
        assert base, f"Can't declare {node.kind.name} outside of {scopes} scope."
        node.offset = base.offset
        node.scope = self
        base.offset += 1
        base.initializers.append(node)
        return True

    def find(self, name: str) -> Optional['DeclNode']:
        if name in self.locals:
            return self.locals[name]
        elif self.parent:
            return self.parent.find(name)
        else:
            return None

    def find_type(self, name: str) -> Optional['TypeNode']:
        module_scope = self.find_scope(NK_MODULE_SCOPE)
        if name in module_scope.types:
            return module_scope.types[name]
        else:
            return module_scope.assembly.types.get(name)

class Parser:
    """Parser class

    Parsing is the second part of the compilation process. It is done ontop
    of a stream of tokens (see `TokenStream` class) that is generated in the
    previous part of the process.

    The parser is made of a wide variety mid to small methods that basically
    performs 2 things:

    * ensure the syntax is valid;
    * construct the AST that will be travesed in the next part of the process.

    Its main method is the `parse()` method, that calls the lexer to produce
    the token stream to be parsed. At the end, a topmost `ScopeNode` object will
    store the finished AST for the source file.

    :param fname: the source code file path to be parsed."""

    def __init__(self, fname: str):
        self.fname: str = fname
        self.ir: List[str] = []
        self.ast: Optional[ASTNode] = None
        self.glb: Dict[str, ASTNode] = {}
        self.scope: ScopeNode = ScopeNode(NK_MODULE_SCOPE)

    def parse(self, assembly_node: AssemblyNode):
        """Starts the `Lexer` and builds the AST from its token stream.

        :returns: None
        """
        assembly_node.modules.append(self.scope)
        self.scope.assembly = assembly_node
        lexer = Lexer()
        stream = lexer.gen_tokens(Source.load(self.fname))
        # for t in stream.tokens:
        #     print(t)
        print('parsing...')
        self.parse_scope(self.scope, stream)
        print('parsing finished')

    def parse_scope(self, scope: ScopeNode, stream: TokenStream) -> None:
        """Parses a block of code.

        The scope being parsed must be instantiated by the caller.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: None
        """
        while True:
            if stream.is_keyword(KW_VARIABLE):
                self.parse_decl_var(scope, stream)

            elif stream.is_keyword(KW_FUNCTION):
                assert scope.parent is None, "Functions must be declared on top-level scope."
                scope.code.append(self.parse_decl_function(scope, stream))

            elif stream.is_keyword(KW_IF):
                scope.code.append(self.parse_stmt_if(scope, stream))

            elif stream.is_keyword(KW_WHILE):
                scope.code.append(self.parse_stmt_while(scope, stream))

            elif stream.is_keyword(KW_DO):
                scope.code.append(self.parse_stmt_do(scope, stream))

            elif stream.is_keyword(KW_REPEAT):
                scope.code.append(self.parse_stmt_repeat(scope, stream))

            elif stream.is_keyword(KW_PRINT):
                scope.code.append(self.parse_stmt_print(scope, stream))

            elif stream.is_keyword(KW_RETURN):
                assert scope.find_scope(NK_FUNCTION_SCOPE), "Return statement is ouside of function."
                scope.code.append(self.parse_stmt_return(scope, stream))

            elif stream.is_keyword(KW_BREAK):
                assert scope.find_scope(NK_LOOP_SCOPE, NK_CASE_SCOPE), "Return statement is ouside of function."
                scope.code.append(self.parse_stmt_break(scope, stream))

            elif stream.is_keyword(KW_CONTINUE):
                assert scope.find_scope(NK_LOOP_SCOPE, NK_CASE_SCOPE), "Return statement is ouside of function."
                scope.code.append(self.parse_stmt_continue(scope, stream))

            elif stream.is_token(TT_NAME):
                scope.code.append(self.parse_stmt_assign(scope, stream))

            elif stream.is_any_operator(OP_INC, OP_DEC):
                scope.code.append(self.parse_stmt_incr(scope, stream))

            else:
                return

    @staticmethod
    def parse_name(stream: TokenStream) -> str:
        """Parses a token of type name.

        It operates on certainty, raising an error if the token is not
        of expected type.

        :param stream: the source stream of tokens.
        :returns: the token value, a string.
        """
        name = stream.token_val
        stream.expect(TT_NAME)
        return name

    @staticmethod
    def parse_literal(token_kind: str, value: str) -> ExprNode:
        value = value.lower()
        suffix = value.strip(SCN_DECDIGITS + TT_DOT)
        if suffix:
            if suffix == SFX_INT8:
                value_range = NK_INT8_EXPR
                value_type = AssemblyNode.types[TY_INT8]
                val = int(value.replace(suffix, ''), 10)
            elif suffix == SFX_INT16:
                value_range = NK_INT16_EXPR
                value_type = AssemblyNode.types[TY_INT16]
                val = int(value.replace(suffix, ''), 10)
            elif suffix == SFX_INT32:
                value_range = NK_INT32_EXPR
                value_type = AssemblyNode.types[TY_INT32]
                val = int(value.replace(suffix, ''), 10)
            elif suffix == SFX_INT64:
                value_range = NK_INT64_EXPR
                value_type = AssemblyNode.types[TY_INT64]
                val = int(value.replace(suffix, ''), 10)
            elif suffix == SFX_UINT8:
                value_range = NK_UINT8_EXPR
                value_type = AssemblyNode.types[TY_UINT8]
                val = int(value.replace(suffix, ''), 10)
            elif suffix == SFX_UINT16:
                value_range = NK_UINT16_EXPR
                value_type = AssemblyNode.types[TY_UINT16]
                val = int(value.replace(suffix, ''), 10)
            elif suffix == SFX_UINT32:
                value_range = NK_UINT32_EXPR
                value_type = AssemblyNode.types[TY_UINT32]
                val = int(value.replace(suffix, ''), 10)
            elif suffix == SFX_UINT64:
                value_range = NK_UINT64_EXPR
                value_type = AssemblyNode.types[TY_UINT64]
                val = int(value.replace(suffix, ''), 10)
            elif suffix == SFX_FLOAT16:
                value_range = NK_FLOAT16_EXPR
                value_type = AssemblyNode.types[TY_FLOAT16]
                val = float(value.replace(suffix, ''))
            elif suffix == SFX_FLOAT32:
                value_range = NK_FLOAT32_EXPR
                value_type = AssemblyNode.types[TY_FLOAT32]
                val = float(value.replace(suffix, ''))
            elif suffix == SFX_FLOAT64:
                value_range = NK_FLOAT64_EXPR
                value_type = AssemblyNode.types[TY_FLOAT64]
                val = float(value.replace(suffix, ''))
            elif suffix == SFX_FLOAT80:
                value_range = NK_FLOAT80_EXPR
                value_type = AssemblyNode.types[TY_FLOAT80]
                val = float(value.replace(suffix, ''))
            else:
                assert False, f"Invalid suffix '{suffix}'"
        else:
            if TT_DOT in value:
                value_range = NK_FLOAT32_EXPR
                value_type = AssemblyNode.types[TY_FLOAT32]
                val = float(value.replace(suffix, ''))
            else:
                value_range = NK_INT32_EXPR
                value_type = AssemblyNode.types[TY_INT32]
                val = int(value.replace(suffix, ''), 10)

        if token_kind == TT_INT:
            error_msg = "Value out of range"
            if value_range == NK_INT8_EXPR:
                assert val <= 127, error_msg

            elif value_range == NK_UINT8_EXPR:
                assert val <= 255, error_msg

            elif value_range == NK_INT16_EXPR:
                assert val <= 32767, error_msg

            elif value_range == NK_UINT16_EXPR:
                assert val <= 65535, error_msg

            elif value_range == NK_INT32_EXPR:
                assert val <= 2147483647, error_msg

            elif value_range == NK_UINT32_EXPR:
                assert val <= 4294967295, error_msg

            elif value_range == NK_INT64_EXPR:
                assert val <= 9223372036854775807, error_msg

            elif value_range == NK_UINT64_EXPR:
                assert val <= 18446744073709551615, error_msg

            else:
                assert False, "Invalid value range."

        return ExprNode(value_range, value=val, constant=val, type=value_type)

    def parse_expr_operand(self, scope: ScopeNode, stream: TokenStream) -> ExprNode:
        """Parses a expression operand.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the operand node.
        """
        token = stream.token_val
        if stream.is_token(TT_INT, TT_FLOAT):
            return self.parse_literal(stream.get_kind(), token)

        elif stream.match_token(TT_NAME):
            node: DeclNode = scope.find(token)
            assert node, f"Undefined name '{token}'"
            node.read()
            node_kind = DECL_TO_EXPR.get(node.kind)
            return ExprNode(node_kind, type=node.type, decl=node)

        else:
            stream.unexpected(TT_INT, TT_FLOAT, TT_NAME)

    def parse_expr_base(self, scope: ScopeNode, stream: TokenStream) -> ExprNode:
        """Parses a expression base operand.

        Calls, subscriptions, lookups are captured here.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the base operand node.
        """
        expr = self.parse_expr_operand(scope, stream)

        if stream.is_token(TT_LPAREN):
            assert expr.kind is NK_FUNCTION_EXPR, f"{expr.kind} cannot be called as a function."
            decl: DeclNode = expr['decl']
            args: List[ExprNode] = self.parse_expr_arguments(scope, decl, stream)
            name = decl.name
            nparams = len(decl.params)
            nargs = len(args)
            assert nparams == nargs, f"'{name}' function expects {nparams} arguments, but {nargs} were passed."
            return ExprNode(NK_FCALL_EXPR, type=decl.type.result, args=args, argc=nargs, decl=decl)
        elif stream.is_any_operator(OP_INC, OP_DEC):
            op = stream.get_val()
            kind = NK_INC_EXPR if op == OP_INC else NK_DEC_EXPR
            what = 'Incrementing' if op == OP_INC else 'Decrementing'
            assert expr.kind not in NK_LITERALS, f"{what} a literal is not OK."
            assert expr.type.accepts(scope.find_type(TY_INT8 if expr.type.signed else TY_UINT8))
            return ExprNode(kind, pre=False, operand=expr)
        return expr

    def parse_expr_arguments(self, scope: ScopeNode, func_node: DeclNode, stream: TokenStream) -> List[ExprNode]:
        """Parser a function's list of arguments.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the list of arguments.
        """
        func_type: TypeNode = func_node.type
        func_params: List[TypeNode] = func_type.params
        nparams: int = len(func_params)
        nargs: int = 0
        args: List[ExprNode] = []
        stream.expect(TT_LPAREN)
        if not stream.is_token(TT_RPAREN):
            arg = self.parse_expr(scope, stream)
            args.append(arg)
            nargs = 1
            while stream.match_token(TT_COMMA):
                arg = self.parse_expr(scope, stream)
                args.append(arg)
                nargs += 1
        assert nargs == nparams, f"'{func_node.name}' expects {nparams} arguments but {nargs} were passed."

        for i, arg in enumerate(args):
            param = func_params[i]
            msg = f"Argument {i} of '{func_node.name}' expects {param.name} value but received a {arg.type.name}."
            assert param.accepts(arg.type), msg

        stream.expect(TT_RPAREN)
        return args

    def parse_expr_unary(self, scope: ScopeNode, stream: TokenStream) -> ExprNode:
        """Parses an unary expression.

        Unary expressions are operations that have only one operand, like
        negation, dereference or increment. It has priority above
        multiplicative expressions.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the expression node.
        """
        if stream.is_any_operator(OP_ADD, OP_SUB, OP_INC, OP_DEC):
            op = stream.get_val()
            if op in (OP_ADD, OP_SUB):
                expr = self.parse_expr_base(scope, stream)
                msg = f"{expr.type.name} does not have sign"
                assert expr.type.kind is NK_PRIMITIVE_TYPE and expr.type.signed, msg
                return ExprNode(NK_UNARY_EXPR, expr=expr.type, op=op, operand=expr)
            elif op in (OP_INC, OP_DEC):
                kind = NK_INC_EXPR if op == OP_INC else NK_DEC_EXPR
                expr = self.parse_expr_base(scope, stream)
                what = 'Incrementing' if op == OP_INC else 'Decrementing'
                assert expr.kind not in NK_LITERALS, f"{what} a literal is not OK."
                assert expr.type.accepts(scope.find_type(TY_INT8 if expr.type.signed else TY_UINT8))
                return ExprNode(kind, type=expr.type, pre=True, operand=expr)
        else:
            return self.parse_expr_base(scope, stream)

    def parse_expr_mul(self, scope: ScopeNode, stream) -> ExprNode:
        """Parses a multiplicative expression.

        These are binary operations like multiplication and division. It
        has priority above addictive expressions.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the expression node.
        """
        expr = self.parse_expr_unary(scope, stream)
        while stream.is_token(TT_MULT, TT_DIV, TT_MOD):
            op = stream.get_val()
            right = self.parse_expr_unary(scope, stream)
            assert expr.type.accepts(right.type), f"Incompatible operand types in {op} expression."
            return ExprNode(NK_BINARY_EXPR, type=expr.type, op=op, left=expr, right=right)
        return expr

    def parse_expr_add(self, scope: ScopeNode, stream) -> ExprNode:
        """Parses a multiplicative expression.

        These are binary operations like addiction and subtraction. It
        has lowest priority.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the expression node.
        """
        expr = self.parse_expr_mul(scope, stream)
        while stream.is_token(TT_PLUS, TT_MINUS):
            op = stream.get_val()
            right = self.parse_expr_mul(scope, stream)
            operands = f"({expr.type.name} and {right.type.name})"
            assert expr.type.accepts(right.type), f"Incompatible operand types {operands} in {op} expression."
            return ExprNode(NK_BINARY_EXPR, type=expr.type, op=op, left=expr, right=right)
        return expr

    def parse_expr_comp(self, scope: ScopeNode, stream: TokenStream) -> ExprNode:
        """Parses a multiplicative expression.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the expression node.
        """
        expr = self.parse_expr_add(scope, stream)
        op = stream.token_val
        while stream.match_any_operator(OP_EQ, OP_NE, OP_LT, OP_LE, OP_GE, OP_GT):
            other = self.parse_expr_add(scope, stream)
            expr_type = scope.find_type(TY_BOOL)
            assert expr.type.accepts(other.type), f"Incompatible operand types in {op} expression."
            expr = ExprNode(NK_COMPARISON_EXPR, type=expr_type, op=op, left=expr, right=other)

        return expr

    def parse_expr_and(self, scope: ScopeNode, stream: TokenStream) -> ExprNode:
        """Parses a logic AND expression.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the expression node.
        """
        expr = self.parse_expr_comp(scope, stream)
        while stream.match_keyword(KW_AND):
            other = self.parse_expr_comp(scope, stream)
            expr_type = scope.find_type(TY_BOOL)
            assert expr.type.accepts(other.type), "Incompatible operand types in AND expression."
            expr = ExprNode(NK_LOGIC_EXPR, type=expr_type, op=KW_AND, left=expr, right=other)

        return expr

    def parse_expr_or(self, scope: ScopeNode, stream: TokenStream) -> ExprNode:
        """Parses a logic OR expression.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the expression node.
        """
        expr = self.parse_expr_and(scope, stream)
        while stream.match_keyword(KW_OR):
            other = self.parse_expr_and(scope, stream)
            expr_type = scope.find_type(TY_BOOL)
            assert expr.type.accepts(other.type), "Incompatible operand types in OR expression."
            expr = ExprNode(NK_LOGIC_EXPR, type=expr_type, op=KW_OR, left=expr, right=other)

        return expr

    def parse_expr_binary(self, scope: ScopeNode, stream: TokenStream) -> ExprNode:
        return self.parse_expr_or(scope, stream)

    def parse_expr_ternary(self, scope: ScopeNode, stream: TokenStream) -> ExprNode:
        expr = self.parse_expr_binary(scope, stream)
        if stream.match_operator(OP_TER):
            then_expr = self.parse_expr(scope, stream)
            stream.expect(TT_COLON)
            else_expr = self.parse_expr(scope, stream)

            assert then_expr.type.accepts(else_expr.type), "Incompatible operand types in ternary expression."

            nodes = {
                'cond_expr': expr,
                'then_expr': then_expr,
                'else_expr': else_expr
            }
            expr = ExprNode(NK_TERNARY_EXPR, type=then_expr.type, **nodes)

        return expr

    def parse_expr(self, scope: ScopeNode, stream: TokenStream) -> ExprNode:
        """Parses an expression.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the expression node.
        """
        return self.parse_expr_ternary(scope, stream)

    def parse_paren_expr(self, scope: ScopeNode, stream: TokenStream) -> ExprNode:
        """Parses an expression enclosed by parenthesis.

        These are binary operations like multiplication and division. It
        has priority above addictive expressions.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the expression node.
        """
        stream.expect(TT_LPAREN)
        expr = self.parse_expr(scope, stream)
        stream.expect(TT_RPAREN)
        return expr

    def parse_decl(self, scope: ScopeNode, stream: TokenStream) -> DeclNode:
        """Parses a declaration.

        Declarations are generally definition of identifiers that will hold
        data in a given context (or scope) or the entry point for executable
        code (like functions and methods).

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the declaration node.
        """
        if stream.is_keyword(KW_VARIABLE):
            return self.parse_decl_var(scope, stream)

        elif stream.is_keyword(KW_FUNCTION):
            assert scope.base_scope is NK_MODULE_SCOPE, "Functions must be declared in module level."
            return self.parse_decl_function(scope, stream)

    def parse_decl_var(self, scope: ScopeNode, stream: TokenStream) -> DeclNode:
        """Parses a variable declaration.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the variable declaration node.
        """
        stream.expect_keyword(KW_VARIABLE)
        decl = self.parse_identifier(scope, stream)
        decl.kind = NK_VAR_DECL
        stream.expect(TT_COLON)
        var_type = self.parse_type_spec(scope, stream)
        # decl = DeclNode(NK_VAR_DECL, name)
        decl.type = var_type
        if stream.match_token(TT_EQUAL):
            expr = self.parse_expr(scope, stream)
            decl.write()
        else:
            expr = ExprNode(NK_UNDEFINED_EXPR)
        decl.initializer = expr
        stream.expect(TT_SEMI)
        assert scope.declare(decl.name, decl, NK_FUNCTION_SCOPE, NK_METHOD_SCOPE), f"{decl.name} already declared."

        return decl

    def parse_decl_function(self, scope: ScopeNode, stream: TokenStream) -> DeclNode:
        """Parses a function declaration.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the function declaration node.
        """
        stream.expect_keyword(KW_FUNCTION)
        name = self.parse_name(stream)
        decl = DeclNode(NK_FUNCTION_DECL, name)
        assert scope.declare(name, decl, NK_MODULE_SCOPE), f"{name} already declared."
        fscope: ScopeNode = ScopeNode(NK_FUNCTION_SCOPE, scope, decl=decl)
        decl.params = self.parse_decl_params(fscope, stream)
        stream.expect(TT_COLON)
        rtype = self.parse_type_spec(scope, stream)
        decl.is_main = name == SW_MAINFUNCTION
        decl.definition = fscope
        ptypes: List[TypeNode] = [p.type for p in decl.params]
        decl.type = TypeNode(NK_FUNCTION_TYPE, name=name, size=rtype.size, result=rtype, params=ptypes)
        stream.expect(TT_LBRACE)
        self.parse_scope(fscope, stream)
        stream.expect(TT_RBRACE)

        fscope.jump_labels.clear()

        return decl

    def parse_decl_params(self, scope: ScopeNode, stream: TokenStream) -> List[DeclNode]:
        """Parses a function's parameter list.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the function's list of parameter nodes.
        """
        params = []
        stream.expect(TT_LPAREN)
        if stream.is_token(TT_NAME):
            params.append(self.parse_decl_param(scope, stream))
            while stream.match_token(TT_COMMA):
                params.append(self.parse_decl_param(scope, stream))

        # invert params offsets
        num_params = len(params)
        for index, param in enumerate(params):
            param.offset = -(num_params - index)
        # set the base frame for variables, reserving space for the frame registers
        scope.offset = scope.FRAME_OFFSET
        stream.expect(TT_RPAREN)
        return params

    def parse_decl_param(self, scope: ScopeNode, stream: TokenStream) -> DeclNode:
        """Parses a parameter declaration.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: a parameter declaration node.
        """
        decl = self.parse_identifier(scope, stream)
        decl.kind = NK_PARAM_DECL
        stream.expect(TT_COLON)
        decl.type = self.parse_type_spec(scope, stream)
        assert scope.declare(decl.name, decl, NK_FUNCTION_SCOPE), f"{decl.name} already declared."
        return decl

    def parse_stmt(self, scope: ScopeNode, stream: TokenStream) -> StmtNode:
        """Parses a statement.

        Statements allows for the manipulation of data, like value assignments,
        execution branching and more.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the statement node.
        """
        if stream.is_keyword(KW_IF):
            return self.parse_stmt_if(scope, stream)

        elif stream.is_keyword(KW_PRINT):
            return self.parse_stmt_print(scope, stream)

        elif stream.is_keyword(KW_RETURN):
            assert scope.find_scope(NK_FUNCTION_SCOPE), "Return statement is ouside of function."
            return self.parse_stmt_return(scope, stream)

    def parse_stmt_if(self, scope: ScopeNode, stream: TokenStream) -> StmtNode:
        """Parses an `if` statement.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the statement node.
        """
        stream.expect_keyword(KW_IF)
        expr = self.parse_paren_expr(scope, stream)
        thenscope: ScopeNode = ScopeNode(NK_STATEMENT_SCOPE, scope)
        stream.expect(TT_LBRACE)
        self.parse_scope(thenscope, stream)
        stream.expect(TT_RBRACE)
        if stream.match_keyword(KW_ELSE):
            elsescope: ScopeNode = ScopeNode(NK_STATEMENT_SCOPE, scope)
            stream.expect(TT_LBRACE)
            self.parse_scope(elsescope, stream)
            stream.expect(TT_RBRACE)
            return StmtNode(NK_IF_ELSE_STMT, expr=expr, thenscope=thenscope, elsescope=elsescope)
        else:
            return StmtNode(NK_IF_THEN_STMT, expr=expr, thenscope=thenscope)

    def parse_stmt_while(self, scope: ScopeNode, stream: TokenStream) -> StmtNode:
        """Parses a `while` statement.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the statement node.
        """
        funcscope: ScopeNode = scope.find_scope(NK_FUNCTION_SCOPE)
        stream.expect_keyword(KW_WHILE)
        expr = self.parse_paren_expr(scope, stream)
        whilescope: ScopeNode = ScopeNode(NK_LOOP_SCOPE, scope)
        label: str = scope.auto_define_label('.enqu')
        if stream.match(TT_COLON):
            label = self.parse_name(stream)
            assert funcscope.define_label(label), f"'{label}' label already declared in this definition"
        funcscope.define_label(label)
        whilescope.set_label(KW_BREAK, label)
        whilescope.set_label(KW_CONTINUE, label)
        stream.expect(TT_LBRACE)
        self.parse_scope(whilescope, stream)
        stream.expect(TT_RBRACE)
        return StmtNode(NK_WHILE_STMT, label=label, expr=expr, whilescope=whilescope)

    def parse_stmt_do(self, scope: ScopeNode, stream: TokenStream) -> StmtNode:
        """Parses a `do` statement.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the statement node.
        """
        funcscope: ScopeNode = scope.find_scope(NK_FUNCTION_SCOPE)
        stream.expect_keyword(KW_DO)
        doscope: ScopeNode = ScopeNode(NK_LOOP_SCOPE, scope)
        label: str = scope.auto_define_label('.faca')
        if stream.match(TT_COLON):
            label = self.parse_name(stream)
            assert funcscope.define_label(label), f"'{label}' label already declared in this definition"
        funcscope.define_label(label)
        doscope.set_label(KW_BREAK, label)
        doscope.set_label(KW_CONTINUE, label)
        stream.expect(TT_LBRACE)
        self.parse_scope(doscope, stream)
        stream.expect(TT_RBRACE)
        kind = NK_DO_WHILE_STMT
        if not stream.match_keyword(KW_WHILE):
            kind = NK_DO_UNTIL_STMT
            stream.expect_keyword(KW_UNTIL)
        expr = self.parse_paren_expr(scope, stream)
        return StmtNode(kind, label=label, expr=expr, doscope=doscope)

    def parse_stmt_repeat(self, scope: ScopeNode, stream: TokenStream) -> StmtNode:
        """Parses a `repeat` statement.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the statement node.
        """
        funcscope: ScopeNode = scope.find_scope(NK_FUNCTION_SCOPE)
        stream.expect_keyword(KW_REPEAT)
        repeatscope: ScopeNode = ScopeNode(NK_LOOP_SCOPE, scope)
        if stream.is_token(TT_LPAREN):
            counter_type = scope.find_type(TY_INT32)
            counter_max = scope.gen_name('max')
            counter_idx = scope.gen_name('idx')
            decl_max = DeclNode(NK_VAR_DECL, counter_max, type=counter_type)
            decl_idx = DeclNode(NK_VAR_DECL, counter_idx, type=counter_type)
            assert repeatscope.declare(counter_max, decl_max, NK_METHOD_SCOPE, NK_FUNCTION_SCOPE)
            assert repeatscope.declare(counter_idx, decl_idx, NK_METHOD_SCOPE, NK_FUNCTION_SCOPE)
            repeatscope.set_iteration(start=counter_idx, stop=counter_max)
            decl_max.initializer = self.parse_paren_expr(scope, stream)
            decl_idx.initializer = ExprNode(NK_INT32_EXPR, type=counter_type, constant=0, value='0')
            repeatscope.code.extend((decl_max, decl_idx))
            kind = NK_REPEAT_FINITE_STMT
        else:
            # expr = ExprNode(NK_UNDEFINED_EXPR)
            kind = NK_REPEAT_INFINITE_STMT
        label: str = scope.auto_define_label('.rept')
        if stream.match(TT_COLON):
            label = self.parse_name(stream)
            assert funcscope.define_label(label), f"'{label}' label already declared in this definition"
        else:
            funcscope.auto_define_label(label)
        repeatscope.set_label(KW_BREAK, label)
        repeatscope.set_label(KW_CONTINUE, label)
        stream.expect(TT_LBRACE)
        self.parse_scope(repeatscope, stream)
        stream.expect(TT_RBRACE)
        return StmtNode(kind, label=label, repeatscope=repeatscope)

    def parse_stmt_break(self, scope: ScopeNode, stream: TokenStream) -> StmtNode:
        """Parses a `break` statement.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the statement node.
        """
        assert scope.find_scope(NK_LOOP_SCOPE, NK_CASE_SCOPE), "Invalid statement."
        funcscope: ScopeNode = scope.find_scope(NK_FUNCTION_SCOPE)
        stream.expect_keyword(KW_BREAK)
        label: Optional[str] = scope.get_label(KW_BREAK)
        if not stream.match(TT_SEMI):
            label = self.parse_name(stream)
            assert funcscope.has_label(label), f"'{label}' label is not declared in this definition"
            stream.expect(TT_SEMI)
        return StmtNode(NK_BREAK_STMT, label=label)

    def parse_stmt_continue(self, scope: ScopeNode, stream: TokenStream) -> StmtNode:
        """Parses a `continue` statement.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the statement node.
        """
        assert scope.find_scope(NK_LOOP_SCOPE), "Invalid statement."
        funcscope: ScopeNode = scope.find_scope(NK_FUNCTION_SCOPE)
        stream.expect_keyword(KW_CONTINUE)
        label: Optional[str] = scope.get_label(KW_CONTINUE)
        if not stream.match(TT_SEMI):
            label = self.parse_name(stream)
            assert funcscope.has_label(label), f"'{label}' label is not declared in this definition"
            stream.expect(TT_SEMI)
        return StmtNode(NK_CONTINUE_STMT, label=label)

    def parse_stmt_print(self, scope: ScopeNode, stream: TokenStream) -> StmtNode:
        """Parses a `print` statement.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the statement node.
        """
        stream.expect_keyword(KW_PRINT)
        expr = self.parse_expr(scope, stream)
        stream.expect(TT_SEMI)
        return StmtNode(NK_PRINT_STMT, expr=expr)

    def parse_stmt_return(self, scope: ScopeNode, stream: TokenStream) -> StmtNode:
        """Parses a `return` statement.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the statement node.
        """
        stream.expect_keyword(KW_RETURN)
        func_scope = scope.find_scope(NK_FUNCTION_SCOPE)
        func_decl = func_scope.decl
        if not stream.is_token(TT_SEMI):
            msg = f"'{func_decl.name}' function has no return value"
            assert func_decl.type.result is not scope.find_type(TY_VOID), msg
            expr = self.parse_expr(scope, stream)
            msg = f"'{func_decl.name}' function should return a value of type {func_decl.type.result.name}"
            assert func_decl.type.result.accepts(expr.type), msg
        else:
            msg = f"'{func_decl.name}' function should return a value of type {func_decl.type.result.name}"
            assert func_decl.type.result is scope.find_type(TY_VOID), msg
            expr = ExprNode(NK_INT32_EXPR, constant=0, value='0')
        stream.expect(TT_SEMI)
        return StmtNode(NK_RETURN_STMT, expr=expr)

    def parse_stmt_assign(self, scope: ScopeNode, stream: TokenStream) -> StmtNode:
        """Parses an assignment statement.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the statement node.
        """
        name = self.parse_name(stream)
        decl = scope.find(name)
        assert decl, f"Undefined name '{name}'"

        if decl and decl.kind is NK_FUNCTION_DECL:
            args: List[ExprNode] = self.parse_expr_arguments(scope, stream)
            name = decl.name
            nparams = len(decl.params)
            nargs = len(args)
            assert nparams == nargs, f"'{name}' function expects {nparams} arguments, but {nargs} were passed."
            stmt = ExprNode(NK_FCALL_EXPR, args=args, argc=nargs, decl=decl)
        else:
            if stream.match_token(TT_EQUAL):
                decl.write()
                expr = self.parse_expr(scope, stream)
                stmt = StmtNode(NK_ASSIGN_STMT, decl=decl, expr=expr)
            elif stream.is_any_operator(OP_INC, OP_DEC):
                op: str = stream.get_val()
                kind = NK_INC_STMT if op == OP_INC else NK_DEC_STMT
                stmt = StmtNode(kind, decl=decl)
            else:
                assert False, "Invalid statement"

        stream.expect(TT_SEMI)
        return stmt

    def parse_stmt_incr(self, scope: ScopeNode, stream: TokenStream) -> StmtNode:
        op: str = stream.get_val()
        expr = self.parse_expr_base(scope, stream)
        kind = NK_INC_STMT if op == OP_INC else NK_DEC_STMT

        stream.expect(TT_SEMI)
        return StmtNode(kind, op=op, expr=expr)

    def parse_type_spec(self, scope: ScopeNode, stream: TokenStream) -> TypeNode:
        module = scope.find_scope(NK_MODULE_SCOPE)
        assert module, "What happened to the module?"
        base_name: str = self.parse_name(stream)

        type_decl = module.assembly.find(base_name)
        assert type_decl, f"Undefined '{base_name}' type."
        return type_decl

    def parse_identifier(self, scope: ScopeNode, stream: TokenStream) -> DeclNode:
        base_type: Optional[TypeNode] = None

        name = self.parse_name(stream)
        while stream.is_token(TT_LBRACKET, TT_MULT):
            if stream.match_token(TT_MULT):
                base_type = TypeNode(NK_POINTER_TYPE, base=base_type)
            else:
                base_type = TypeNode(NK_ARRAY_TYPE, base=base_type)
                if not stream.is_token(TT_RBRACKET):
                    base_type.length = self.parse_expr(scope, stream)
                    stream.expect(TT_RBRACKET)
        decl = DeclNode(NK_NAME, name=name)
        decl.type = base_type
        return decl

# endregion (classes)
# ---------------------------------------------------------
