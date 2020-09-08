import random
from typing import Optional, List, Dict, Tuple, Union

from brah.constants.tokens import *
from brah.constants.nodes import *
from brah.constants.errors import *
from brah.a_scanner import Source, SCN_DECDIGITS
from brah.b_lexer import *
from brah.f_utils import error, assertion, Location

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
    NK_CONSTANT_DECL: NK_CONSTANT_EXPR,
    NK_FUNCTION_DECL: NK_FUNCTION_EXPR
}


# endregion (constants)
# ---------------------------------------------------------
# region FUNCTIONS

def eval_op_result_kind(op: str, l_kind: NodeKind, r_kind: NodeKind) -> Optional[NodeKind]:
    if l_kind is r_kind:
        return l_kind

    if l_kind in NK_SIGNED_EXPR:
        if r_kind in NK_SIGNED_EXPR:
            return l_kind if l_kind > r_kind else r_kind
        elif r_kind in NK_UNSIGNED_EXPR:
            if op in ('<<', '>>'):
                return l_kind
            return None
        elif r_kind in NK_FLOAT_EXPR:
            return r_kind

    elif l_kind in NK_UNSIGNED_EXPR:
        if r_kind in NK_SIGNED_EXPR:
            return None
        elif r_kind in NK_UNSIGNED_EXPR:
            return l_kind if l_kind > r_kind else r_kind
        elif r_kind in NK_FLOAT_EXPR:
            return r_kind

    elif l_kind in NK_FLOAT_EXPR:
        if r_kind in NK_INTEGER_EXPR:
            return l_kind
        elif r_kind in NK_FLOAT_EXPR:
            return l_kind if l_kind > r_kind else r_kind

    else:
        return None


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
        self.decl: Optional[str, DeclNode] = kwargs.get('decl')

    def __getitem__(self, key) -> Union['ExprNode', 'DeclNode', 'ScopeNode']:
        return self.nodes.__getitem__(key)

    def __setitem__(self, key, value: Union['ExprNode', 'DeclNode']):
        self.nodes.__setitem__(key, value)


class ExprNode(ASTNode):

    def __init__(self, kind: NodeKind, **kwargs):
        super(ExprNode, self).__init__(kind, **kwargs)
        self.nodes: Dict[str, Union[str, ExprNode]] = kwargs.copy()
        self.decl: Optional[str, DeclNode] = kwargs.get('decl')
        self.op: str = kwargs.get('op', '')
        self.value: str = kwargs.get('value', '0')
        self.type: Optional[TypeNode] = kwargs.get('type')
        self.constant: Optional[Union[int, float]] = kwargs.get('constant')
        self.is_constexpr: bool = kwargs.get('is_constexpr')

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

        # classes
        self.implements: List[TypeNode] = []

        # customization
        self.interfaces: Dict[str, bool] = {}

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
        # scopes = ", nor outside of ".join(n.name for n in expected_scopes)
        # assert base, f"Can't declare {node.kind.name} outside of {scopes} scope."
        if not base:
            return False
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

    def get_constant(self, expr: ExprNode) -> Union[int, float]:
        if expr.kind is NK_CONSTANT_EXPR:
            return expr.decl.initializer.constant
        elif expr.kind is NK_CONSTEXPR_EXPR or expr.is_constexpr:
            return expr.constant

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
                assertion(scope.find_scope(NK_FUNCTION_SCOPE),
                          stream.token_loc, EK_SCOP, ERR_WRONG_STMT_SCOPE, KW_VARIABLE, 'function definition')
                self.parse_decl_var(scope, stream)

            elif stream.is_keyword(KW_CONSTANT):
                assertion(scope.parent is None, stream.token_loc, EK_SCOP, ERR_WRONG_DECL_SCOPE, 'Functions')
                scope.code.append(self.parse_decl_constant(scope, stream))

            elif stream.is_keyword(KW_FUNCTION):
                assertion(scope.parent is None, stream.token_loc, EK_SCOP, ERR_WRONG_DECL_SCOPE, 'Functions')
                scope.code.append(self.parse_decl_function(scope, stream))

            elif stream.is_keyword(KW_IF):
                assertion(scope.find_scope(NK_FUNCTION_SCOPE),
                          stream.token_loc, EK_SCOP, ERR_WRONG_STMT_SCOPE, KW_IF, 'function definition')
                scope.code.append(self.parse_stmt_if(scope, stream))

            elif stream.is_keyword(KW_WHILE):
                assertion(scope.find_scope(NK_FUNCTION_SCOPE),
                          stream.token_loc, EK_SCOP, ERR_WRONG_STMT_SCOPE, KW_WHILE, 'function definition')
                scope.code.append(self.parse_stmt_while(scope, stream))

            elif stream.is_keyword(KW_DO):
                assertion(scope.find_scope(NK_FUNCTION_SCOPE),
                          stream.token_loc, EK_SCOP, ERR_WRONG_STMT_SCOPE, KW_DO, 'function definition')
                scope.code.append(self.parse_stmt_do(scope, stream))

            elif stream.is_keyword(KW_REPEAT):
                assertion(scope.find_scope(NK_FUNCTION_SCOPE),
                          stream.token_loc, EK_SCOP, ERR_WRONG_STMT_SCOPE, KW_REPEAT, 'function definition')
                scope.code.append(self.parse_stmt_repeat(scope, stream))

            elif stream.is_keyword(KW_PRINT):
                assertion(scope.find_scope(NK_FUNCTION_SCOPE),
                          stream.token_loc, EK_SCOP, ERR_WRONG_STMT_SCOPE, KW_PRINT, 'function definition')
                scope.code.append(self.parse_stmt_print(scope, stream))

            elif stream.is_keyword(KW_RETURN):
                assertion(scope.find_scope(NK_FUNCTION_SCOPE),
                          stream.token_loc, EK_SCOP, ERR_WRONG_STMT_SCOPE, KW_RETURN, 'function definition')
                scope.code.append(self.parse_stmt_return(scope, stream))

            elif stream.is_keyword(KW_BREAK):
                assertion(scope.find_scope(NK_LOOP_SCOPE, NK_CASE_SCOPE),
                          stream.token_loc, EK_SCOP, ERR_WRONG_STMT_SCOPE, KW_BREAK, 'loop or switch statements')
                scope.code.append(self.parse_stmt_break(scope, stream))

            elif stream.is_keyword(KW_CONTINUE):
                assertion(scope.find_scope(NK_LOOP_SCOPE, NK_CASE_SCOPE),
                          stream.token_loc, EK_SCOP, ERR_WRONG_STMT_SCOPE, KW_CONTINUE, 'loop or switch statements')
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
    def parse_literal(location: Location, kind: str, value: str) -> ExprNode:
        value = value.lower()
        value_range = NK_NONE
        value_type = AssemblyNode.types[TY_VOID]
        val = 0
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
                error(location, EK_EXPR, ERR_INVALID_SUFFIX, suffix)
        else:
            if TT_DOT in value:
                value_range = NK_FLOAT32_EXPR
                value_type = AssemblyNode.types[TY_FLOAT32]
                val = float(value.replace(suffix, ''))
            else:
                value_range = NK_INT32_EXPR
                value_type = AssemblyNode.types[TY_INT32]
                val = int(value.replace(suffix, ''), 10)

        if kind == TT_INT:
            if value_range == NK_INT8_EXPR:
                if not val <= 127:
                    error(location, EK_EXPR, ERR_OUT_OF_RNG)

            elif value_range == NK_UINT8_EXPR:
                if not val <= 255:
                    error(location, EK_EXPR, ERR_INVALID_SUFFIX)

            elif value_range == NK_INT16_EXPR:
                if not val <= 32767:
                    error(location, EK_EXPR, ERR_INVALID_SUFFIX)

            elif value_range == NK_UINT16_EXPR:
                if not val <= 65535:
                    error(location, EK_EXPR, ERR_INVALID_SUFFIX)

            elif value_range == NK_INT32_EXPR:
                if not val <= 2147483647:
                    error(location, EK_EXPR, ERR_INVALID_SUFFIX)

            elif value_range == NK_UINT32_EXPR:
                if not val <= 4294967295:
                    error(location, EK_EXPR, ERR_INVALID_SUFFIX)

            elif value_range == NK_INT64_EXPR:
                if not val <= 9223372036854775807:
                    error(location, EK_EXPR, ERR_INVALID_SUFFIX)

            elif value_range == NK_UINT64_EXPR:
                if not val <= 18446744073709551615:
                    error(location, EK_EXPR, ERR_INVALID_SUFFIX)

            elif value_range in (NK_FLOAT16_EXPR, NK_FLOAT32_EXPR, NK_FLOAT64_EXPR, NK_FLOAT80_EXPR):
                pass

            else:
                assertion(False, location, EK_EXPR, ERR_OUT_OF_RNG)

        return ExprNode(value_range, is_constexpr=True, value=str(val), constant=val, type=value_type)

    def parse_expr_operand(self, scope: ScopeNode, stream: TokenStream, const_expr: bool = False) -> ExprNode:
        """Parses a expression operand.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :param const_expr: whether the expression node must be constant.
        :returns: the operand node.
        """
        token = stream.token_val
        if stream.is_token(TT_INT, TT_FLOAT):
            tkn = stream.get()
            expr = self.parse_literal(tkn.location, tkn.kind, token)
            if const_expr:
                expr.kind = NK_CONSTEXPR_EXPR
            return expr

        elif stream.is_token(TT_NAME):
            loc = stream.get().location
            node: DeclNode = scope.find(token)
            assertion(node, loc, EK_DECL, ERR_UNDEFINED_NAME, token)
            if const_expr:
                assertion(node.kind is NK_CONSTANT_DECL,
                          loc, EK_EXPR, ERR_NOT_CONSTEXPR, node.name)
            node.read()
            node_kind = DECL_TO_EXPR.get(node.kind)
            return ExprNode(node_kind, type=node.type, decl=node)

        else:
            stream.unexpected(TT_INT, TT_FLOAT, TT_NAME)

    def parse_expr_base(self, scope: ScopeNode, stream: TokenStream, const_expr: bool = False) -> ExprNode:
        """Parses a expression base operand.

        Calls, subscriptions, lookups are captured here.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :param const_expr: whether the expression node must be constant.
        :returns: the base operand node.
        """
        expr = self.parse_expr_operand(scope, stream, const_expr)

        if stream.is_token(TT_LPAREN):
            assertion(not const_expr,
                      stream.token_loc, EK_EXPR, ERR_CONSTEXPR_CALL)
            assertion(expr.kind is NK_FUNCTION_EXPR,
                      stream.token_loc, EK_EXPR, ERR_NOT_CALLABLE, expr.type.name)
            decl: DeclNode = expr['decl']
            args: List[ExprNode] = self.parse_expr_arguments(scope, decl, stream)
            # name = decl.name
            # nparams = len(decl.params)
            nargs = len(args)
            return ExprNode(NK_FCALL_EXPR, is_constexpr=False, type=decl.type.result, args=args, argc=nargs, decl=decl)
        elif stream.is_any_operator(OP_INC, OP_DEC):
            loc = stream.token_loc
            op = stream.get_val()
            kind = NK_INC_EXPR if op == OP_INC else NK_DEC_EXPR
            what = 'Incrementing' if op == OP_INC else 'Decrementing'
            assertion(expr.kind not in NK_LITERALS,
                      loc, EK_EXPR, ERR_LITERAL_INCR_DECR, what)
            assertion(expr.type.integer,
                      loc, EK_EXPR, ERR_CANNOT_INCR_DECR, what, expr.type.name)
            assertion(not const_expr,
                      loc, EK_EXPR, ERR_CANNOT_INCR_DECR, what, 'constant expression')
            return ExprNode(kind, is_constexpr=expr.is_constexpr, pre=False, operand=expr)
        return expr

    def parse_expr_arguments(self, scope: ScopeNode, func_node: DeclNode, stream: TokenStream) -> List[ExprNode]:
        """Parser a function's list of arguments.

        :param scope: the current scope being parsed.
        :param func_node: the function whose arguments are being passed.
        :param stream: the source stream of tokens.
        :returns: the list of arguments.
        """
        func_type: TypeNode = func_node.type
        func_params: List[TypeNode] = func_type.params
        nparams: int = len(func_params)
        nargs = 0
        args: List[ExprNode] = []
        index: int = 0
        stream.expect(TT_LPAREN)
        if not stream.is_token(TT_RPAREN):
            loc = stream.token_loc
            arg = self.parse_expr(scope, stream)
            args.append(arg)
            nargs += 1
            if nparams:
                assertion(func_params[index].accepts(arg.type), loc, EK_TYPE, ERR_WRONG_ARG_TYPE,
                          index, func_node.name, func_params[index].name, arg.type)
            index += 1
            while stream.match_token(TT_COMMA):
                loc = stream.token_loc
                arg = self.parse_expr(scope, stream)
                args.append(arg)
                nargs += 1
                if nparams > index:
                    assertion(func_params[index].accepts(arg.type),
                              loc, EK_TYPE, ERR_WRONG_ARG_TYPE,
                              index, func_node.name, func_params[index].name, arg.type)
                index += 1

        assertion(nparams == nargs, stream.token_loc, EK_STMT, ERR_WRONG_ARG_NUMBER,
                  func_node.name, nparams, nargs)
        stream.expect(TT_RPAREN)
        return args

    def parse_expr_unary(self, scope: ScopeNode, stream: TokenStream, const_expr: bool = False) -> ExprNode:
        """Parses an unary expression.

        Unary expressions are operations that have only one operand, like
        negation, dereference or increment. It has priority above
        multiplicative expressions.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :param const_expr: whether the expression node must be constant.
        :returns: the expression node.
        """
        if stream.is_any_operator(OP_ADD, OP_SUB, OP_INC, OP_DEC):
            loc = stream.token_loc
            op = stream.get_val()
            if op in (OP_ADD, OP_SUB):
                expr = self.parse_expr_base(scope, stream, const_expr)
                assertion(expr.type.kind is NK_PRIMITIVE_TYPE and expr.type.signed,
                          loc, EK_EXPR, ERR_NOT_SIGNED, expr.type.name)
                if const_expr or expr.is_constexpr:
                    expr.constant = -expr.constant if op == OP_SUB else +expr.constant
                    return expr
                return ExprNode(NK_UNARY_EXPR, is_constexpr=expr.is_constexpr, type=expr.type, op=op, operand=expr)
            elif op in (OP_INC, OP_DEC):
                kind = NK_INC_EXPR if op == OP_INC else NK_DEC_EXPR
                expr = self.parse_expr_base(scope, stream, const_expr)
                what = 'Incrementing' if op == OP_INC else 'Decrementing'
                assertion(expr.kind not in NK_LITERALS,
                          loc, EK_EXPR, ERR_LITERAL_INCR_DECR, what)
                assertion(expr.type.integer,
                          loc, EK_EXPR, ERR_CANNOT_INCR_DECR, what, expr.type.name)
                assertion(not const_expr,
                          loc, EK_EXPR, ERR_CANNOT_INCR_DECR, what, 'constant expression')
                return ExprNode(kind, is_constexpr=False, type=expr.type, pre=True, operand=expr)
        else:
            return self.parse_expr_base(scope, stream, const_expr)

    def parse_expr_mul(self, scope: ScopeNode, stream, const_expr: bool = False) -> ExprNode:
        """Parses a multiplicative expression.

        These are binary operations like multiplication and division. It
        has priority above addictive expressions.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :param const_expr: whether the expression node must be constant.
        :returns: the expression node.
        """
        expr = self.parse_expr_unary(scope, stream, const_expr)
        while stream.is_token(TT_MULT, TT_DIV, TT_MOD):
            loc = stream.token_loc
            op = stream.get_val()
            right = self.parse_expr_unary(scope, stream, const_expr)
            operands = f"({expr.type.name} and {right.type.name})"
            assertion(expr.type.accepts(right.type), loc, EK_TYPE, ERR_INCOMPATIBLE_TYPES, operands, op)
            if const_expr or (expr.is_constexpr and right.is_constexpr):
                kind: Optional[NodeKind] = eval_op_result_kind(op, expr.kind, right.kind)
                l_op = self.get_constant(expr)
                r_op = self.get_constant(right)
                if op == TT_MULT:
                    val = l_op * r_op
                elif op == TT_DIV:
                    assertion(l_op != 0, loc, EK_EXPR, ERR_DIVISION_BY_ZERO)
                    if expr.type.integer:
                        val = l_op // r_op
                    else:
                        val = l_op / r_op
                else:  # op == TT_MOD:
                    val = l_op % r_op
                expr = ExprNode(kind, is_constexpr=True, type=expr.type, constant=val, value=str(val))
            else:
                expr = ExprNode(NK_BINARY_EXPR, type=expr.type, op=op, left=expr, right=right)
        return expr

    def parse_expr_add(self, scope: ScopeNode, stream, const_expr: bool = False) -> ExprNode:
        """Parses a multiplicative expression.

        These are binary operations like addiction and subtraction. It
        has lowest priority.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :param const_expr: whether the expression node must be constant.
        :returns: the expression node.
        """
        expr = self.parse_expr_mul(scope, stream, const_expr)
        while stream.is_token(TT_PLUS, TT_MINUS):
            loc = stream.token_loc
            op = stream.get_val()
            right = self.parse_expr_mul(scope, stream, const_expr)
            operands = f"({expr.type.name} and {right.type.name})"
            assertion(expr.type.accepts(right.type), loc, EK_TYPE, ERR_INCOMPATIBLE_TYPES, operands, op)
            if const_expr or (expr.is_constexpr and right.is_constexpr):
                kind: Optional[NodeKind] = eval_op_result_kind(op, expr.kind, right.kind)
                l_op = self.get_constant(expr)
                r_op = self.get_constant(right)
                if op == TT_PLUS:
                    val = l_op + r_op
                else:  # op == TT_MINUS:
                    val = l_op - r_op
                expr = ExprNode(kind, is_constexpr=True, type=expr.type, constant=val, value=str(val))
            else:
                expr = ExprNode(NK_BINARY_EXPR, type=expr.type, op=op, left=expr, right=right)
        return expr

    def parse_expr_comp(self, scope: ScopeNode, stream: TokenStream, const_expr: bool = False) -> ExprNode:
        """Parses a multiplicative expression.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :param const_expr: whether the expression node must be constant.
        :returns: the expression node.
        """
        expr = self.parse_expr_add(scope, stream, const_expr)
        while stream.is_any_operator(OP_EQ, OP_NE, OP_LT, OP_LE, OP_GE, OP_GT):
            loc = stream.token_loc
            op = stream.get_val()
            right = self.parse_expr_add(scope, stream, const_expr)
            expr_type = scope.find_type(TY_BOOL)
            operands = f"({expr.type.name} and {right.type.name})"
            assertion(expr.type.accepts(right.type), loc, EK_TYPE, ERR_INCOMPATIBLE_TYPES, operands, op)
            if const_expr:
                val = expr.constant
                if op == OP_EQ:
                    val = val == right.constant
                if op == OP_NE:
                    val = val != right.constant
                if op == OP_LT:
                    val = val < right.constant
                if op == OP_LE:
                    val = val <= right.constant
                if op == OP_GE:
                    val = val > right.constant
                else:  # op == OP_GT:
                    val = val >= right.constant
                expr = ExprNode(NK_CONSTEXPR_EXPR, expr_type, constant=val, value=int(val))
            else:
                expr = ExprNode(NK_COMPARISON_EXPR, type=expr_type, op=op, left=expr, right=right)

        return expr

    def parse_expr_and(self, scope: ScopeNode, stream: TokenStream, const_expr: bool = False) -> ExprNode:
        """Parses a logic AND expression.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :param const_expr: whether the expression node must be constant.
        :returns: the expression node.
        """
        expr = self.parse_expr_comp(scope, stream, const_expr)
        while stream.match_keyword(KW_AND):
            other = self.parse_expr_comp(scope, stream, const_expr)
            expr_type = scope.find_type(TY_BOOL)
            if const_expr:
                val = expr.constant and other.constant
                expr = ExprNode(NK_CONSTEXPR_EXPR, expr_type, constant=val, value=int(val))
            else:
                expr = ExprNode(NK_LOGIC_EXPR, type=expr_type, op=KW_AND, left=expr, right=other)

        return expr

    def parse_expr_or(self, scope: ScopeNode, stream: TokenStream, const_expr: bool = False) -> ExprNode:
        """Parses a logic [X]OR expression.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :param const_expr: whether the expression node must be constant.
        :returns: the expression node.
        """
        expr = self.parse_expr_and(scope, stream, const_expr)
        kw = stream.token_val
        while stream.match_any_keyword(KW_OR, KW_XOR):
            other = self.parse_expr_and(scope, stream, const_expr)
            expr_type = scope.find_type(TY_BOOL)
            if const_expr:
                if kw == KW_OR:
                    val = expr.constant or other.constant
                else:
                    val = (expr.constant and not other.constant) or (not expr.constant and other.constant)
                expr = ExprNode(NK_CONSTEXPR_EXPR, expr_type, constant=val, value=int(val))
            else:
                expr = ExprNode(NK_LOGIC_EXPR, type=expr_type, op=KW_OR, left=expr, right=other)

        return expr

    def parse_expr_binary(self, scope: ScopeNode, stream: TokenStream, const_expr: bool = False) -> ExprNode:
        return self.parse_expr_or(scope, stream, const_expr)

    def parse_expr_ternary(self, scope: ScopeNode, stream: TokenStream, const_expr: bool = False) -> ExprNode:
        expr = self.parse_expr_binary(scope, stream, const_expr)
        loc = stream.token_loc
        if stream.match_operator(OP_TER):
            assertion(not const_expr,
                      loc, EK_EXPR, ERR_CONSTEXPR_TERNARY)
            then_expr = self.parse_expr(scope, stream)
            stream.expect(TT_COLON)
            else_expr = self.parse_expr(scope, stream)

            operands = f"({else_expr.type.name} and {then_expr.type.name})"
            assertion(then_expr.type.accepts(else_expr.type), loc, EK_TYPE, ERR_INCOMPATIBLE_TYPES, operands, 'ternary')

            nodes = {
                'cond_expr': expr,
                'then_expr': then_expr,
                'else_expr': else_expr
            }
            expr = ExprNode(NK_TERNARY_EXPR, type=then_expr.type, **nodes)

        return expr

    def parse_expr(self, scope: ScopeNode, stream: TokenStream, const_expr: bool = False) -> ExprNode:
        """Parses an expression.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :param const_expr: whether the expression node must be constant.
        :returns: the expression node.
        """
        return self.parse_expr_ternary(scope, stream, const_expr)

    def parse_paren_expr(self, scope: ScopeNode, stream: TokenStream, const_expr: bool = False) -> ExprNode:
        """Parses an expression enclosed by parenthesis.

        These are binary operations like multiplication and division. It
        has priority above addictive expressions.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :param const_expr: whether the expression node must be constant.
        :returns: the expression node.
        """
        stream.expect(TT_LPAREN)
        expr = self.parse_expr(scope, stream, const_expr)
        stream.expect(TT_RPAREN)
        return expr

    def parse_decl_var(self, scope: ScopeNode, stream: TokenStream) -> DeclNode:
        """Parses a variable declaration.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the variable declaration node.
        """
        stream.expect_keyword(KW_VARIABLE)
        loc = stream.token_loc
        decl = self.parse_identifier(scope, stream)
        decl.kind = NK_VAR_DECL
        stream.expect(TT_COLON)
        var_type = self.parse_type_spec(scope, stream)
        decl.type = var_type
        if stream.match_token(TT_EQUAL):
            expr = self.parse_expr(scope, stream)
            decl.write()
        else:
            expr = ExprNode(NK_UNDEFINED_EXPR)
        decl.initializer = expr
        stream.expect(TT_SEMI)
        declared = scope.declare(decl.name, decl, NK_FUNCTION_SCOPE, NK_METHOD_SCOPE)
        assertion(declared, loc, EK_DECL, ERR_REDECLARED_NAME, decl.name)

        return decl

    def parse_decl_constant(self, scope: ScopeNode, stream: TokenStream) -> DeclNode:
        """Parses a function declaration.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the function declaration node.
        """
        stream.expect_keyword(KW_CONSTANT)
        loc = stream.token_loc
        name = self.parse_name(stream)
        decl = DeclNode(NK_CONSTANT_DECL, name)
        declared = scope.declare(name, decl, NK_MODULE_SCOPE)
        assertion(declared, loc, EK_DECL, ERR_REDECLARED_NAME, name)
        stream.expect(TT_COLON)
        decl.type = self.parse_type_spec(scope, stream)
        stream.expect(TT_EQUAL)
        const_expr: ExprNode = self.parse_expr(scope, stream, True)
        expr = self.parse_literal(loc, TT_INT if isinstance(const_expr.constant, int) else TT_FLOAT,
                                  str(const_expr.constant))
        decl.initializer = expr
        stream.expect(TT_SEMI)

        return decl

    def parse_decl_function(self, scope: ScopeNode, stream: TokenStream) -> DeclNode:
        """Parses a function declaration.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the function declaration node.
        """
        stream.expect_keyword(KW_FUNCTION)
        loc = stream.token_loc
        name = self.parse_name(stream)
        decl = DeclNode(NK_FUNCTION_DECL, name)
        declared = scope.declare(name, decl, NK_MODULE_SCOPE)
        assertion(declared, loc, EK_DECL, ERR_REDECLARED_NAME, name)
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
        loc = stream.token_loc
        decl = self.parse_identifier(scope, stream)
        decl.kind = NK_PARAM_DECL
        stream.expect(TT_COLON)
        decl.type = self.parse_type_spec(scope, stream)
        declared = scope.declare(decl.name, decl, NK_FUNCTION_SCOPE)
        assertion(declared, loc, EK_DECL, ERR_REDECLARED_NAME, decl.name)
        return decl

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
            loc = stream.token_loc
            label = self.parse_name(stream)
            defined = funcscope.define_label(label)
            assertion(defined, loc, EK_DECL, ERR_REDECLARED_LABEL, label)
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
            loc = stream.token_loc
            label = self.parse_name(stream)
            defined = funcscope.define_label(label)
            assertion(defined, loc, EK_DECL, ERR_REDECLARED_LABEL, label)
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
            repeatscope.declare(counter_max, decl_max, NK_METHOD_SCOPE, NK_FUNCTION_SCOPE)
            repeatscope.declare(counter_idx, decl_idx, NK_METHOD_SCOPE, NK_FUNCTION_SCOPE)
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
            loc = stream.token_loc
            label = self.parse_name(stream)
            defined = funcscope.define_label(label)
            assertion(defined, loc, EK_DECL, ERR_REDECLARED_LABEL, label)
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
        funcscope: ScopeNode = scope.find_scope(NK_FUNCTION_SCOPE)
        stream.expect_keyword(KW_BREAK)
        label: Optional[str] = scope.get_label(KW_BREAK)
        if not stream.match(TT_SEMI):
            loc = stream.token_loc
            label = self.parse_name(stream)
            defined = funcscope.has_label(label)
            assertion(defined, loc, EK_DECL, ERR_REDECLARED_LABEL, label)
            stream.expect(TT_SEMI)
        return StmtNode(NK_BREAK_STMT, label=label)

    def parse_stmt_continue(self, scope: ScopeNode, stream: TokenStream) -> StmtNode:
        """Parses a `continue` statement.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the statement node.
        """
        funcscope: ScopeNode = scope.find_scope(NK_FUNCTION_SCOPE)
        stream.expect_keyword(KW_CONTINUE)
        label: Optional[str] = scope.get_label(KW_CONTINUE)
        if not stream.match(TT_SEMI):
            loc = stream.token_loc
            label = self.parse_name(stream)
            defined = funcscope.has_label(label)
            assertion(defined, loc, EK_DECL, ERR_REDECLARED_LABEL, label)
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
            loc = stream.token_loc
            assertion(func_decl.type.result is not scope.find_type(TY_VOID),
                      stream.token.location, EK_DECL, ERR_NO_VOID_RETURN, func_decl.name)
            expr = self.parse_expr(scope, stream)
            assertion(func_decl.type.result.accepts(expr.type),
                      loc, EK_TYPE, ERR_WRONG_RETURN_TYPE, func_decl.name, func_decl.type.result.name)
        else:
            assertion(func_decl.type.result is scope.find_type(TY_VOID),
                      stream.token.location, EK_DECL, ERR_VOID_RETURN, func_decl.name, func_decl.type.result.name)
            expr = ExprNode(NK_INT32_EXPR, constant=0, value='0')
        stream.expect(TT_SEMI)
        return StmtNode(NK_RETURN_STMT, expr=expr)

    def parse_stmt_assign(self, scope: ScopeNode, stream: TokenStream) -> StmtNode:
        """Parses an assignment statement.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the statement node.
        """
        loc = stream.token_loc
        name = self.parse_name(stream)
        decl = scope.find(name)
        assertion(decl, loc, EK_DECL, ERR_UNDEFINED_NAME, name)

        if decl and decl.kind is NK_FUNCTION_DECL:
            args: List[ExprNode] = self.parse_expr_arguments(scope, stream)
            name = decl.name
            nparams = len(decl.params)
            nargs = len(args)
            assertion(nparams == nargs, loc, EK_STMT, ERR_WRONG_ARG_NUMBER, name, nparams, nargs)
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
                assertion(False, stream.token_loc, EK_STMT, ERR_INVALID_STMT)
                assert False

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
        assert module, "This is an error related to the language implementation."
        loc = stream.token_loc
        base_name: str = self.parse_name(stream)

        type_decl = module.assembly.find(base_name)
        assertion(type_decl, loc, EK_DECL, ERR_UNDEFINED_NAME, base_name)
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
