"""
Implements a recursive descent parser for the smpl language that returns a 
abstract syntax tree.

Author: André Rösti

The EBNF specification of the smpl language is reproduced here for convenience:

letter = “a”|“b”|...|“z”.
digit = “0”|“1”|...|“9”.
relOp = “==“|“!=“|“<“|“<=“|“>“|“>=“.

ident = letter {letter | digit}.
number = digit {digit}.

designator = ident{ "[" expression "]" }.
factor = designator | number | “(“ expression “)” | funcCall . 
term = factor { (“*” | “/”) factor}.
expression = term {(“+” | “-”) term}.
relation = expression relOp expression .

assignment = “let” designator “<-” expression.
funcCall = “call” ident [ “(“ [expression { “,” expression } ] “)” ]. 
ifStatement = “if” relation “then” statSequence [ “else” statSequence ] “fi”. 
whileStatement = “while” relation “do” StatSequence “od”. 
returnStatement = “return” [ expression ] .

statement = assignment | funcCall | ifStatement | whileStatement | returnStatement.
statSequence = statement { “;” statement } [ “;” ] .
typeDecl = “var” | “array” “[“ number “]” { “[“ number “]” }.
varDecl = typeDecl indent { “,” ident } “;” .
funcDecl = [ “void” ] “function” ident formalParam “;” funcBody “;” . 
formalParam = “(“ [ident { “,” ident }] “)” .
funcBody = { varDecl } “{” [ statSequence ] “}”.
computation = “main” { varDecl } { funcDecl } “{” statSequence “}” “.” .
"""

import re
import ast


class SmplToken:
    IDENT     = re.compile(r"[a-zA-Z]([a-zA-Z0-9]+)?")
    NUMBER    = re.compile(r"[0-9]+")
    OP_INEQ   = '!='
    OP_EQ     = '=='
    OP_LT     = '<'
    OP_LE     = '<='
    OP_GT     = '>'
    OP_GE     = '>='
    LBRACKET  = '['
    RBRACKET  = ']'
    LPAREN    = '('
    RPAREN    = ')'
    LBRACE    = '{'
    RBRACE    = '}'
    ASTERISK  = '*'
    SLASH     = '/'
    PLUS      = '+'
    MINUS     = '-'
    LET       = 'let'
    LARROW    = '<-'
    CALL      = 'call'
    IF        = 'if'
    THEN      = 'then'
    ELSE      = 'else'
    FI        = 'fi'
    WHILE     = 'while'
    DO        = 'do'
    OD        = 'od'
    RETURN    = 'return'
    VAR       = 'var'
    ARRAY     = 'array'
    SEMICOLON = ';'
    VOID      = re.compile(r'void\s+')
    FUNCTION  = re.compile(r'function\s+')
    MAIN      = re.compile(r'main\s+')
    PERIOD    = '.'
    COMMA     = ','

    # List of all tokens in order of precedence
    # Note that the longest match has highest precedence, only after that this
    # order is considered.
    tokens = [IDENT,     NUMBER,    OP_INEQ,   OP_EQ,     OP_LT,     OP_LE,     
              OP_GT,     OP_GE,     LBRACKET,  RBRACKET,  LPAREN,    RPAREN,    
              LBRACE,    RBRACE,    ASTERISK,  SLASH,     PLUS,      MINUS,    
              LET,        
              LARROW,    CALL,      IF,        THEN,      ELSE,      FI,        
              WHILE,     
              DO,        OD,        RETURN,    VAR,       ARRAY,     SEMICOLON, 
              VOID,      FUNCTION,  MAIN,      PERIOD,    COMMA    ]    
    tokens.reverse()

    def __init__(self, token, val=None, pos=None, line=None, col=None):
        self.token = token
        self.val = val
        self.pos = pos
        self.line = line
        self.col = col


class SmplLexer:
    
    def __init__(self, instring):
        self.instring = instring.strip()
        self.pos = 0
        self.current = (0, None)
        self.next()

    def tokenize(self, pos):
        substr = self.instring[pos:]
        line = self.instring[:pos].count("\n") + 1
        col = pos - self.instring[:pos].rfind("\n")
        if not substr:
            return (0, None)  # end of string reached
        n_consumed, token = 0, None
        for candidate in SmplToken.tokens:  # check in order of precedence
            this_n_consumed, this_token = 0, None
            if isinstance(candidate, re.Pattern):
                match = candidate.match(substr)
                if not match:
                    continue
                this_token = SmplToken(candidate, val=match.group(0), pos=pos, line=line, col=col)
                this_n_consumed = match.end(0)
            else:
                if not substr.startswith(candidate):
                    continue
                this_token = SmplToken(candidate, pos=pos, line=line, col=col)
                this_n_consumed = len(candidate)
            if n_consumed < this_n_consumed:  # longest match wins (precedence)
                n_consumed = this_n_consumed
                token = this_token
        if not n_consumed:
            raise Exception("Lexer: no matching pattern at line {0:d}, col {1:d}: {2:s}".format(line, col, self.instring[pos]))
        whitespace = 0
        pos += n_consumed
        while pos < len(self.instring) and self.instring[pos].isspace():
            pos += 1
            whitespace += 1  # advance past and ignore whitespace
        n_consumed += whitespace
        return (n_consumed, token)

    def peek(self):
        """
        Return token at current cursor position without consuming it.
        "Look ahead."
        """
        _, token = self.current
        return token

    def next(self):
        """
        Consume and return current token, parse the next one in the stream, if
        any.
        """
        consumed_len, consumed = self.current
        self.pos += consumed_len
        self.current = self.tokenize(self.pos)  # parse next token into self.upnext
        return consumed


class SmplParseError(Exception):
    pass


class SmplParseWarning(Exception):
    pass


class SmplParser:

    factor_terminals = {SmplToken.IDENT, SmplToken.NUMBER, SmplToken.LPAREN, SmplToken.CALL}
    statement_terminals = {SmplToken.LET, SmplToken.CALL, SmplToken.IF, SmplToken.WHILE, SmplToken.RETURN}

    def __init__(self, inlexer):
        self.inlexer = inlexer
        self.current = 0

    def parse(self):
        if self.current != 0:
            raise Exception("Parser is single-use.")
        tree = ast.AST(self.computation())
        return tree

    def _consume(self, tokens, error=True, warn=False):
        consumed = self.inlexer.next()
        if consumed and consumed.token in tokens:
            return consumed
        expected_tokens = ", ".join([t.pattern if isinstance(t, re.Pattern) else t for t in tokens])
        text = "Unexpected end of file, expected one of {}".format(expected_tokens)
        if consumed:
            msg = "expected one of {0:s}, got {1:s}"
            consumed_token = consumed.token.pattern if isinstance(consumed.token, re.Pattern) else consumed.token
            text = ("line {0:d}, column {1:d} (char {2:d}):".format(consumed.line, consumed.col, consumed.pos) +
                    msg.format(expected_tokens, consumed_token))
        if error:
            raise SmplParseError(text)
        if warn:
            raise SmplParseWarning(text)
        return consumed

    def _peek(self, tokens):
        nextup = self.inlexer.peek()
        return nextup.token in tokens

    def computation(self):
        vdecls = []
        fdecls = []
        self._consume({SmplToken.MAIN})
        while self._peek({SmplToken.VAR, SmplToken.ARRAY}):
            vdecl = self.var_decl()
            vdecls.extend(vdecl)
        while self._peek({SmplToken.VOID, SmplToken.FUNCTION}):
            fdecl = self.func_decl()
            fdecls.append(fdecl)
        self._consume({SmplToken.LBRACE})
        stmts = self.stat_sequence()
        self._consume({SmplToken.RBRACE})
        self._consume({SmplToken.PERIOD})
        return ast.Computation(vdecls, fdecls, stmts)

    def var_decl(self):
        dims = self.type_decl()
        idents = [self.ident()]
        while self._peek({SmplToken.COMMA}):
            self._consume({SmplToken.COMMA})
            idents.append(self.ident())
        self._consume({SmplToken.SEMICOLON})
        vdecls = [ast.VariableDeclaration(ident, dims) for ident in idents]
        return vdecls

    def type_decl(self):
        """
        We only support integer types, and arrays of integers. Hence, it
        suffices that this parser returns a list of array dimensions. For a
        scalar, an empty list is returned.
        """
        consumed = self._consume({SmplToken.VAR, SmplToken.ARRAY})
        dims = None  # dims == None indicates scalar value
        if consumed.token == SmplToken.ARRAY:
            dims = []
            self._consume({SmplToken.LBRACKET})
            n = self.number()
            dims.append(n)
            self._consume({SmplToken.RBRACKET})
            while self._peek({SmplToken.LBRACKET}):
                self._consume({SmplToken.LBRACKET})
                n = self.number()
                dims.append(n)
                self._consume({SmplToken.RBRACKET})
        return dims

    def number(self):
        consumed = self._consume({SmplToken.NUMBER})
        return ast.Number(int(consumed.val))

    def func_decl(self):
        is_void = False
        if self._peek({SmplToken.VOID}):
            is_void = True
            self._consume({SmplToken.VOID})
        self._consume({SmplToken.FUNCTION})
        ident = self.ident()
        param_idents = self.formal_param()
        self._consume({SmplToken.SEMICOLON})
        local_vdecls, stmts = self.func_body()
        self._consume({SmplToken.SEMICOLON})
        return ast.FunctionDeclaration(ident, param_idents, local_vdecls, stmts, is_void)
        
    def ident(self):
        letters = self._consume({SmplToken.IDENT})
        identifier = letters.val
        return ast.Identifier(identifier)

    def formal_param(self):
        self._consume({SmplToken.LPAREN})
        param_idents = []
        if self._peek({SmplToken.IDENT}):
            param_idents.append(self.ident())
            while self._peek({SmplToken.COMMA}):
                self._consume({SmplToken.COMMA})
                param_idents.append(self.ident())
        self._consume({SmplToken.RPAREN})
        return param_idents

    def func_body(self):
        local_vdecls = []
        while self._peek({SmplToken.VAR, SmplToken.ARRAY}):
            local_vdecls.extend(self.var_decl())
        self._consume({SmplToken.LBRACE})
        stmts = []
        if self._peek(self.statement_terminals):
            stmts = self.stat_sequence()
        self._consume({SmplToken.RBRACE})
        return local_vdecls, stmts

    def stat_sequence(self):
        stmt = self.statement()
        stmts = [stmt]
        while self._peek({SmplToken.SEMICOLON}):
            self._consume({SmplToken.SEMICOLON})
            if self._peek(self.statement_terminals):
                stmts.append(self.statement())
        return stmts
    
    def statement(self):
        if self._peek({SmplToken.LET}):
            return self.assignment()
        elif self._peek({SmplToken.CALL}):
            return self.func_call()
        elif self._peek({SmplToken.IF}):
            return self.if_statement()
        elif self._peek({SmplToken.WHILE}):
            return self.while_statement()
        elif self._peek({SmplToken.RETURN}):
            return self.return_statement()
        else:
            self._consume(self.statement_terminals)
            # Only using the consume function here to produce the error message;
            # if one of the valid tokens is actually present, they will be
            # consumed in the respective recursive calls

    def assignment(self):
        self._consume({SmplToken.LET})
        lhs = self.designator()
        self._consume({SmplToken.LARROW})
        rhs = self.expression()
        return ast.Assignment(lhs, rhs)

    def designator(self):
        ident = self.ident()
        indices = []
        while self._peek({SmplToken.LBRACKET}):
            self._consume({SmplToken.LBRACKET})
            indices.append(self.expression())
            self._consume({SmplToken.RBRACKET})
        if not indices:  # scalar access
            return ident
        return ast.ArrayAccess(ident, indices)

    def expression(self):
        opa = self.term()
        while self._peek({SmplToken.PLUS, SmplToken.MINUS}):
            op_tkn = self._consume({SmplToken.PLUS, SmplToken.MINUS})
            op = "+" if op_tkn.token == SmplToken.PLUS else "-"
            opb = self.term()
            opa = ast.BinOp(op, opa, opb)
        return opa

    def term(self):
        opa = self.factor()
        while self._peek({SmplToken.ASTERISK, SmplToken.SLASH}):
            op_tkn = self._consume({SmplToken.ASTERISK, SmplToken.SLASH})
            op = "*" if op_tkn.token == SmplToken.ASTERISK else "/"
            opb = self.factor()
            opa = ast.BinOp(op, opa, opb)
        return opa

    def factor(self):
        ret = None
        if self._peek({SmplToken.IDENT}):
            ret = self.designator()
        elif self._peek({SmplToken.NUMBER}):
            ret = self.number()
        elif self._peek({SmplToken.LPAREN}):
            self._consume({SmplToken.LPAREN})
            ret = self.expression()
            self._consume({SmplToken.RPAREN})
        elif self._peek({SmplToken.CALL}):
            ret = self.func_call()
        else:
            self._consume(self.factor_terminals)
            # only for error message
        return ret
    
    def relation(self):
        opa = self.expression()
        rel_op_tkn = self.rel_op()
        opb = self.expression()
        op = ("==" if rel_op_tkn.token == SmplToken.OP_EQ else
              "!=" if rel_op_tkn.token == SmplToken.OP_INEQ else
              "<"  if rel_op_tkn.token == SmplToken.OP_LT else
              "<=" if rel_op_tkn.token == SmplToken.OP_LE else
              ">"  if rel_op_tkn.token == SmplToken.OP_GT else
              ">=" if rel_op_tkn.token == SmplToken.OP_GE else None)
        return ast.BinOp(op, opa, opb)

    def rel_op(self):
        return self._consume({SmplToken.OP_EQ, SmplToken.OP_INEQ, SmplToken.OP_LT,
                              SmplToken.OP_LE, SmplToken.OP_GT, SmplToken.OP_GE})

    def func_call(self):
        self._consume({SmplToken.CALL})
        func_ident = self.ident()
        params = []
        if self._peek({SmplToken.LPAREN}):
            self._consume({SmplToken.LPAREN})
            if self._peek(self.factor_terminals):
                params.append(self.expression())
                while self._peek({SmplToken.COMMA}):
                    self._consume({SmplToken.COMMA})
                    params.append(self.expression())
            self._consume({SmplToken.RPAREN})
        return ast.FuncCall(func_ident, params)

    def if_statement(self):
        self._consume({SmplToken.IF})
        condition = self.relation()
        self._consume({SmplToken.THEN})
        stmts = self.stat_sequence()
        else_stmts = []
        if self._peek({SmplToken.ELSE}):
            self._consume({SmplToken.ELSE})
            else_stmts = self.stat_sequence()
        self._consume({SmplToken.FI})
        return ast.IfStatement(condition, stmts, else_stmts)

    def while_statement(self):
        self._consume({SmplToken.WHILE})
        condition = self.relation()
        self._consume({SmplToken.DO})
        stmts = self.stat_sequence()
        self._consume({SmplToken.OD})
        return ast.WhileStatement(condition, stmts)

    def return_statement(self):
        self._consume({SmplToken.RETURN})
        val = None
        if self._peek(self.factor_terminals):
            val = self.expression()
        return ast.ReturnStatement(val)

    
