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
            raise Exception("Lexer: no matching pattern at char {0:d}: {1:s}".format(pos, self.instring[pos]))
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

    def _consume(self, tokens, msg="expected one of {0:s}, got {1:s}", error=True, warn=False):
        consumed = self.inlexer.next()
        if consumed.token in tokens:
            return consumed
        msg = "expected one of {0:s}, got {1:s}"
        expected_tokens = ", ".join([t.pattern if isinstance(t, re.Pattern) else t for t in tokens])
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
        self._consume({SmplToken.MAIN}, "smpl computation must start with keyword main")
        while self._peek({SmplToken.VAR, SmplToken.ARRAY}):
            vdecl = self.var_decl()
            vdecls.append(vdecl)
        while self._peek({SmplToken.VOID, SmplToken.FUNCTION}):
            fdecl = self.func_decl()
            fdecls.append(fdecl)
        self._consume({SmplToken.LBRACE})
        stmts = self.stat_sequence()
        self._consume({SmplToken.RBRACE})
        self._consume({SmplToken.PERIOD})
        return ast.Computation(vdecls, fdecls, stmts)

    def var_decl(self):
        tdecl = self.type_decl()
        ident = self.ident()
        idents = [ident]
        while self._peek({SmplToken.COMMA}):
            self._consume({SmplToken.COMMA})
            ident = self.ident()
            idents.append(ident)
        self._consume({SmplToken.SEMICOLON})
        vdecls = [ast.VarDecl(tdecl, ident) for ident in idents]
        return vdecls

    def type_decl(self):
        """
        We only support integer types, and arrays of integers. Hence, it
        suffices that this parser returns a list of array dimensions. For a
        scalar, an empty list is returned.
        """
        consumed = self._consume({SmplToken.VAR, SmplToken.ARRAY})
        dims = []
        if consumed.token == SmplToken.ARRAY:
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
        return int(consumed.val)

    def func_decl(self):
        isvoid = False
        if self._peek({SmplToken.VOID}):
            isvoid = True
            self._consume({SmplToken.VOID})
        self._consume({SmplToken.FUNCTION})
        self.ident()
        self.formal_param()
        self._consume({SmplToken.SEMICOLON})
        self.func_body()
        self._consume({SmplToken.SEMICOLON})
        return ast.FuncDecl()
        
    def ident(self):
        letters = self._consume({SmplToken.IDENT})
        identifier = letters.val
        return identifier

    def formal_param(self):
        self._consume({SmplToken.LPAREN})
        if self._peek({SmplToken.IDENT}):
            self.ident()
            while self._peek({SmplToken.COMMA}):
                self._consume({SmplToken.COMMA})
                self.ident()
        self._consume({SmplToken.RPAREN})

    def func_body(self):
        while self._peek({SmplToken.VAR, SmplToken.ARRAY}):
            self.var_decl()
        self._consume({SmplToken.LBRACE})
        if self._peek(self._statement_terminals):
            self.stat_sequence()
        self._consume({SmplToken.RBRACE})

    def stat_sequence(self):
        stmt = self.statement()
        stmts = [stmt]
        while self._peek({SmplToken.SEMICOLON}):
            self._consume({SmplToken.SEMICOLON})
            stmt = self.statement()
            stmts.append(stmt)
        if self._peek({SmplToken.SEMICOLON}):
            self._consume({SmplToken.SEMICOLON})
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
        self.designator()
        self._consume({SmplToken.LARROW})
        self.expression()
        return ast.AssignmentNode()

    def designator(self):
        self.ident()
        while self._peek({SmplToken.LBRACKET}):
            self._consume({SmplToken.LBRACKET})
            self.expression()
            self._consume({SmplToken.RBRACKET})

    def expression(self):
        self.term()
        while self._peek({SmplToken.PLUS, SmplToken.MINUS}):
            self._consume({SmplToken.PLUS, SmplToken.MINUS})
            self.term()

    def term(self):
        self.factor()
        while self._peek({SmplToken.ASTERISK, SmplToken.SLASH}):
            self._consume({SmplToken.ASTERISK, SmplToken.SLASH})
            self.factor()

    def factor(self):
        if self._peek({SmplToken.IDENT}):
            self.designator()
        elif self._peek({SmplToken.NUMBER}):
            self.number()
        elif self._peek({SmplToken.LPAREN}):
            self._consume({SmplToken.LPAREN})
            self.expression()
            self._consume({SmplToken.RPAREN})
        elif self._peek({SmplToken.CALL}):
            self.func_call()
        else:
            self._consume(self.factor_terminals)
            # only for error message
    
    def relation(self):
        self.expression()
        self.rel_op()
        self.expression()

    def rel_op(self):
        self._consume({SmplToken.OP_EQ, SmplToken.OP_INEQ, SmplToken.OP_LT,
                       SmplToken.OP_LE, SmplToken.OP_GT, SmplToken.OP_GE})

    def func_call(self):
        self._consume({SmplToken.CALL})
        self.ident()
        if self._peek({SmplToken.LPAREN}):
            self._consume({SmplToken.LPAREN})
            if self._peek(self.factor_terminals):
                self.expression()
                while self._peek({SmplToken.COMMA}):
                    self._consume({SmplToken.COMMA})
                    self.expression()
            self._consume({SmplToken.RPAREN})
        return ast.CallNode()

    def if_statement(self):
        self._consume({SmplToken.IF})
        self.relation()
        self._consume({SmplToken.THEN})
        stmts = self.stat_sequence()
        if self._peek({SmplToken.ELSE}):
            self._consume({SmplToken.ELSE})
            else_stmts = self.stat_sequence()
            stmts.extend(else_stmts)
        self._consume({SmplToken.FI})
        return ast.IfNode(stmts)

    def while_statement(self):
        self._consume({SmplToken.WHILE})
        self.relation()
        self._consume({SmplToken.DO})
        stmts = self.stat_sequence()
        self._consume({SmplToken.OD})
        return ast.WhileNode(stmts)

    def return_statement(self):
        self._consume({SmplToken.RETURN})
        if self._peek(self.factor_terminals):
            self.expression()
        return ast.ReturnNode()

    
