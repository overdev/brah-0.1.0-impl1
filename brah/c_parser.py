from typing import Optional, Any, List, Dict, NamedTuple, Tuple, Union
from brah.constants.tokens import *
from brah.constants.nodes import *
from brah.constants.scopes import *
from brah.b_lexer import *
from brah.a_scanner import Source

__all__ = [
    'AstNode',
    'Parser',
    'Scope',
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


class Scope:
    """Scope class.

    Defines a local execution context where names (variables and parameters)
    can be defined, and statements can be executed. It is basically a block
    of code.

    Scopes can be nested and this allows for name lookups in preceding
    contexts. It is an error to look for a name that doesn't exist (i.e.
    wich is not defined).

    Basic control of name access (read and write) is given. Each access
    increments a counter for the reading or writing made. This allows for
    unused name detection.

    :param offset: the base ofset for the local names in the scope.
    :param parent: the optional parent scope.
    """

    FRAME_OFFSET = 3

    def __init__(self, offset: int = 0, parent: Optional['Scope'] = None, scope_kind: ScopeKind = SK_STATEMENT):
        self.parent: Optional['Scope'] = parent
        self.names: Dict[str, AstNode] = {}
        self.base_offset: int = offset
        self.block: AstNode = AstNode(NK_BLOCK, {'code': [], 'scope': self})
        self.kind: ScopeKind = scope_kind

    @property
    def is_module_level(self):
        return self.parent and self.parent.kind is SK_MODULE

    def find(self, *scope_kinds: ScopeKind) -> Optional['Scope']:
        """Performs a scope lookup and returns the first scope of given kinds.

        Useful to find, for example, the scope of a function being defined.

        :param scope_kinds: The kinds of scope to match.
        :returns: The scope found, or None
        """
        if self.kind in scope_kinds:
            return self
        elif self.parent:
            return self.parent.find(*scope_kinds)
        else:
            return None

    def declare_entrypoint(self, name: str, node: 'AstNode') -> None:
        """Declares a new name (it must be a function) in the current scope.

        :param name: the variable or parameter name.
        :param node: the AST node that defines de name.
        :returns: None.
        """
        assert self.is_module_level, "Entry function must be defined in module level."
        assert node.kind is NK_FUNCTION_DECL, f"'{name}' is not a function"
        self.declare(name, node)

    def declare(self, name: str, node: 'AstNode') -> None:
        """Declares a new name (variable or parameter) in the current scope.

        :param name: the variable or parameter name.
        :param node: the AST node that defines de name.
        :returns: None.
        """
        assert name not in self.names, f"'{name}' already defined in this scope."
        node.data['offset'] = self.base_offset
        self.base_offset += 1
        self.names[name] = node

    def read(self, name: str) -> Optional['AstNode']:
        """Perform a name lookup and returns the first node matching the name or None.

        If the node is found in the current scope, its read counter is incremented
        and the node is returned. Alternatively, if the scope has a parent scope,
        that scope is looked up. The lookup chain continues until the topmost scope
        return a node or None.

        :param name: the name to find.
        :returns: The AST node found, or None.
        """
        if name in self.names:
            self.names[name].data['reads'] += 1
            return self.names[name]
        elif self.parent:
            return self.parent.read(name)
        else:
            return None

    def write(self, name: str) -> bool:
        """Perform a name lookup and returns the first node matching the name or None.

        If the node is found in the current scope, its write counter is incremented
        and the node is returned. Alternatively, if the scope has a parent scope,
        that scope is looked up. The lookup chain continues until the topmost scope
        return a node or None.

        :param name: the name to find.
        :returns: The AST node found, or None.
        """
        if name in self.names:
            self.names[name].data['writes'] += 1
            return True
        elif self.parent:
            return self.parent.write(name)
        else:
            return False

    def get(self, name: str) -> Optional['AstNode']:
        """Perform a name lookup and returns the first node matching the name or None.

        Unlike `read()` and `write()`, it does not increment access counters.

        :param name: the name to find.
        :returns: The AST node found, or None.
        """
        if name in self.names:
            return self.names[name]
        elif self.parent:
            return self.parent.get(name)
        else:
            return None


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


class AssemblyNode(ASTNode):

    def __init__(self, kind: NodeKind):
        # assert kind in (NK_MODULE,)
        super(AssemblyNode, self).__init__(kind)
        self.modules: List[AssemblyNode] = []


class DeclNode(ASTNode):

    def __init__(self, kind: NodeKind, name: str):
        super(DeclNode, self).__init__(kind)
        self.name: str = name
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

    def __getitem__(self, key) -> Union['ExprNode', 'DeclNode', 'ScopeNode']:
        return self.nodes.__getitem__(key)

    def __setitem__(self, key, value: Union['ExprNode', 'DeclNode']):
        self.nodes.__setitem__(key, value)


class ExprNode(ASTNode):

    def __init__(self, kind: NodeKind, **kwargs):
        super(ExprNode, self).__init__(kind, **kwargs)
        self.nodes: Dict[str, Union[str, ExprNode]] = kwargs.copy()
        self.op: str = kwargs.get('op', '')

    def __getitem__(self, key) -> Union['ExprNode', 'DeclNode']:
        return self.nodes.__getitem__(key)

    def __setitem__(self, key, value: Union['ExprNode', 'DeclNode']):
        self.nodes.__setitem__(key, value)


class TypeNode(ASTNode):

    def __init__(self, kind: NodeKind, **kwargs):
        super(TypeNode, self).__init__(kind, **kwargs)


class ScopeNode(ASTNode):

    FRAME_OFFSET = 3

    def __init__(self, kind: NodeKind, parent: Optional['ScopeNode'] = None):
        super(ScopeNode, self).__init__(kind)
        self.parent: Optional[ScopeNode] = parent
        self.locals: Dict[str, DeclNode] = {}
        self.code: List[Union[StmtNode, DeclNode]] = []
        self.offset: int = 0

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
        if self.kind is NK_STATEMENT_SCOPE:
            assert self.parent is not None, "Fatal Error: STATEMENT_SCOPE not parented by another scope"
            return self.parent.base_scope
        else:
            return self

    def find_scope(self, kind: NodeKind) -> Optional['ScopeNode']:
        if self.kind is kind:
            return self
        elif self.parent:
            return self.parent.find_scope(kind)
        else:
            return None

    def declare(self, name: str, node: DeclNode) -> bool:
        if name in self.locals:
            return False

        self.locals[name] = node
        base = self.base_scope
        node.offset = base.offset
        node.scope = self
        base.offset += 1
        return True

    def find(self, name: str) -> Optional['DeclNode']:
        if name in self.locals:
            return self.locals[name]
        elif self.parent:
            return self.parent.find(name)
        else:
            return None


class AstNode(NamedTuple):
    """AstNode (Abstract Syntax Tree Node) named tuple class.

    Defines a single node in the Abstract Syntax Tree structure.

    An `AstNode` instance's only purpose is to hold minimum necessary
    information that allows it to be compiled later into bytecode. The
    information saved depends on the kind of node, but it is mainly
    separated in two members: `data` and `nodes`.

    In data, information related to the node itself is stored. It might be,
    for example, the string of a variable's name.

    In nodes, any child node that is structurally connected to this node is
    stored. It might be, for example, the left and right operands of a
    binary operation. It's optional.

    **Members:**

    `kind`: the kind that distinguishes this node, a `NodeKind` enum
    member.

    `data`: the node's own aditional information.

    `nodes`: the optional child nodes this node might have.
    """
    kind: NodeKind
    data: Dict[str, Any]
    nodes: Optional[Dict[str, 'AstNode']] = None

    def __repr__(self):
        data = ', '.join([f"{k}: {v}" for (k, v) in self.data.items()]) if self.data else ''
        nodes = ', '.join([f"{k}: {v}" for (k, v) in self.nodes.items()]) if self.nodes else ''
        return f"{self.kind.name}({data}, {nodes})"

    __str__ = __repr__


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
        self.ast: Optional[AstNode] = None
        self.glb: Dict[str, AstNode] = {}
        self.scope: ScopeNode = ScopeNode(NK_MODULE_SCOPE)

    def parse(self):
        """Starts the `Lexer` and builds the AST from its token stream.

        :returns: None
        """
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

            elif stream.is_some_keyword(KW_IF, KW_PRINT, KW_RETURN):
                scope.code.append(self.parse_stmt(scope, stream))

            elif stream.is_token(TT_NAME):
                scope.code.append(self.parse_stmt_assign(scope, stream))

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
    def parse_expr_operand(scope: ScopeNode, stream: TokenStream) -> ExprNode:
        """Parses a expression operand.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the operand node.
        """
        token = stream.token_val
        if stream.match_token(TT_INT):
            return ExprNode(NK_INT32_EXPR, value=token)

        elif stream.match_token(TT_FLOAT):
            return ExprNode(NK_INT64_EXPR, value=token)

        elif stream.match_token(TT_NAME):
            node: DeclNode = scope.find(token)
            assert node, f"Undefined name '{token}'"
            node.read()
            node_kind = DECL_TO_EXPR.get(node.kind)
            return ExprNode(node_kind, decl=node)

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
            args: List[ExprNode] = self.parse_expr_arguments(scope, stream)
            name = decl.name
            nparams = len(decl.params)
            nargs = len(args)
            assert nparams == nargs, f"'{name}' function expects {nparams} arguments, but {nargs} were passed."
            return ExprNode(NK_FCALL_EXPR, args=args, argc=nargs, decl=decl)
        return expr

    def parse_expr_arguments(self, scope: ScopeNode, stream: TokenStream) -> List[ExprNode]:
        """Parser a function's list of arguments.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the list of arguments.
        """
        args = []
        stream.expect(TT_LPAREN)
        if not stream.is_token(TT_RPAREN):
            args.append(self.parse_expr(scope, stream))
            while stream.match_token(TT_COMMA):
                args.append(self.parse_expr(scope, stream))
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
        op = stream.token_val
        if stream.match_token(TT_PLUS, TT_MINUS):
            return ExprNode(NK_UNARY_EXPR, op=op, operand=self.parse_expr_base(scope, stream))
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
            op = stream.token_val
            stream.next()
            return ExprNode(NK_BINARY_EXPR, op=op, left=expr, right=self.parse_expr_unary(scope, stream))
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
            op = stream.token_val
            stream.next()
            return ExprNode(NK_BINARY_EXPR, op=op, left=expr, right=self.parse_expr_mul(scope, stream))
        return expr

    def parse_expr_comp(self, scope: ScopeNode, stream: TokenStream) -> ExprNode:
        """Parses a multiplicative expression.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the expression node.
        """
        expr = self.parse_expr_add(scope, stream)
        op = stream.token_val
        while stream.match_some_operator(OP_EQ, OP_NE, OP_LT, OP_LE, OP_GE, OP_GT):
            other = self.parse_expr_add(scope, stream)
            expr = ExprNode(NK_COMPARISON_EXPR, op=op, left=expr, right=other)

        return expr

    def parse_expr_and(self, scope: ScopeNode, stream: TokenStream) -> ExprNode:
        """Parses a multiplicative expression.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the expression node.
        """
        expr = self.parse_expr_comp(scope, stream)
        while stream.match_keyword(KW_AND):
            other = self.parse_expr_comp(scope, stream)
            expr = ExprNode(NK_LOGIC_EXPR, op=KW_AND, left=expr, right=other)

        return expr

    def parse_expr_or(self, scope: ScopeNode, stream: TokenStream) -> ExprNode:
        """Parses a ? expression.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the expression node.
        """
        expr = self.parse_expr_and(scope, stream)
        while stream.match_keyword(KW_OR):
            other = self.parse_expr_and(scope, stream)
            expr = ExprNode(NK_LOGIC_EXPR, op=KW_OR, left=expr, right=other)

        return expr

    def parse_expr_binary(self, scope: ScopeNode, stream: TokenStream) -> ExprNode:
        return self.parse_expr_or(scope, stream)

    def parse_expr_ternary(self, scope: ScopeNode, stream: TokenStream) -> ExprNode:
        expr = self.parse_expr_binary(scope, stream)
        if stream.match_operator(OP_TER):
            then_expr = self.parse_expr(scope, stream)
            stream.expect(TT_COLON)
            else_expr = self.parse_expr(scope, stream)

            nodes = {
                'cond_expr': expr,
                'then_expr': then_expr,
                'else_expr': else_expr
            }
            expr = ExprNode(NK_TERNARY_EXPR, **nodes)

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
        name = self.parse_name(stream)
        decl = DeclNode(NK_VAR_DECL, name)
        if stream.match_token(TT_EQUAL):
            expr = self.parse_expr(scope, stream)
            decl.write()
        else:
            expr = ExprNode(NK_UNDEFINED_EXPR)
        decl.initializer = expr
        stream.expect(TT_SEMI)
        assert scope.declare(name, decl), f"{name} already declared."

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
        assert scope.declare(name, decl), f"{name} already declared."
        fscope: ScopeNode = ScopeNode(NK_FUNCTION_SCOPE, scope)
        decl.params = self.parse_decl_params(fscope, stream)
        decl.is_main = name == SW_MAINFUNCTION
        decl.definition = fscope
        stream.expect(TT_LBRACE)
        self.parse_scope(fscope, stream)
        stream.expect(TT_RBRACE)

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
        name = self.parse_name(stream)
        decl = DeclNode(NK_PARAM_DECL, name)
        assert scope.declare(name, decl), f"{name} already declared."
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

    def parse_stmt_print(self, scope: ScopeNode, stream: TokenStream) -> StmtNode:
        """Parses a print statement.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the statement node.
        """
        stream.expect_keyword(KW_PRINT)
        expr = self.parse_expr(scope, stream)
        stream.expect(TT_SEMI)
        return StmtNode(NK_PRINT_STMT, expr=expr)

    def parse_stmt_return(self, scope: ScopeNode, stream: TokenStream) -> StmtNode:
        """Parses a return statement.

        :param scope: the current scope being parsed.
        :param stream: the source stream of tokens.
        :returns: the statement node.
        """
        stream.expect_keyword(KW_RETURN)
        if not stream.is_token(TT_SEMI):
            expr = self.parse_expr(scope, stream)
        else:
            expr = ExprNode(NK_INT32_EXPR, value='0')
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
            assert stream.match_token(TT_EQUAL), "Invalid statement"
            decl.write()
            expr = self.parse_expr(scope, stream)
            stmt = StmtNode(NK_ASSIGN_STMT, decl=decl, expr=expr)
        stream.expect(TT_SEMI)
        return stmt


# endregion (classes)
# ---------------------------------------------------------
