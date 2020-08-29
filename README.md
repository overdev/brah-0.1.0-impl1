# BRAH Programming Language

## EBNF Grammar Specification

```
Module       = Imports Declarations
Imports      = Import Imports
Import       = "importe" "{" Names "}" "de" SOURCEFILE ";"
Names        = NAME [ "," Names ]
Declarations = Declaration Declarations
Declaration  = Constant
             | Function
             | Structure
             | Interface
             | Class
Constant     = "constante" NAME "=" Expression ";"
Variable     = "var" NAME [ "=" Expression ] ";"
ProtOrDef    = ( ";" | Definition )
Definition   = ( FunDef | StrucDef | IntefDef | ClassDef )
Function     = "função" NAME "(" [ Parameters ] ")" ProtOrDef
Structure    = "estrutura" NAME ProtOrDef
Interface    = "interface" NAME ProtOrDef
Class        = "classe" NAME [ "extende" NAME ] [ "implementa" NAME ] ProtOrDef



```

### Function Definition

**Keywords**

* KW_RETURN (`retorne`)
* KW_FUNCTION (`função`)

**Nodes**

* NK_FUNCTION_DECL
* NK_PARAM_DECL
* NK_RETURN_STMT
* NK_PARAM_EXPR
* NK_FCALL_EXPR
* NK_FUNCTION_EXPR

**Mnemonics**

* CALL codePtr numArgs
* RET
* JMP

**Parser**

* `parse_decl_function(self, scope: Scope, stream: TokenStream) -> ASTNode`
* `parse_decl_fblock(self, scope: Scope, stream: TokenStream) -> ASTNode`
* `parse_decl_params(self, scope: Scope, stream: TokenStream) -> ASTNode`
* `parse_decl_param(self, scope: Scope, stream: TokenStream) -> ASTNode`
* `parse_stmt_return(self, scope: Scope, stream: TokenStream) -> ASTNode`

**Compiler**

* `compile_decl_function(self, decl_node: AstNode) -> None`
* `compile_expr_fcall(self, decl_node: AstNode) -> None`
* `compile_stmt_return(self, decl_node: AstNode) -> None`


---

### `If` Statement

**Keywords**

* KW_IF (`se`)
* KW_ELSE (`senão`)
* KW_AND (`e`)
* KW_OR (`ou`)
* KW_OUX (`oux`)
* KW_IS (`é`)
* KW_NOT (`não`)

**Nodes**

* NK_IF_THEN_STMT
* NK_IF_ELSE_STMT
* NK_TERNARY_EXPR
* NK_LOGIC_EXPR
* NK_COMPARISSON_EXPR

**Mnemonics**

* CMP
* JE
* JNE
* JA
* JAE
* JB
* JBE
* JZ
* JNZ
* AND
* ANDL
* OR
* ORL
* XOR
* XORL
* NOT
* NOTL

**Operators**

* Equal (`==`)
* Not Equal (`!=`)
* Greater Than (`>`)
* Greater Than or Equal (`>=`)
* Lesser Than (`<`)
* Lesser Than or Equal (`<=`)
* Binary And (`&`)
* Binary Or (`|`)
* Binary Xor (`^`)
* Binary Not (`~`)
* Ternary (`?`)

**Parser**

`parse_stmt_if(self, scope: Scope, stream: TokenStream) -> AstNode`
`parse_expr_ternary(self, scope: Scope, stream: TokenStream) -> AstNode`

**Compiler**

`compile_stmt_ifthen() -> None`
`compile_stmt_ifelse() -> None`
`compile_expr_ternary() -> None`
