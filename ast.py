"""
Abstract syntax tree nodes for the Smpl language. Provides methods that convert
AST nodes to SSA (single static assignment) instruction streams for compilation
and also methods for evaluation (interpreter).
"""


class ASTnode:

    def __init__(self, children=[], parent=None, val=None, label=None):
        self.children = children 
        self.parent = parent
        self.label = label
        self.val = None

    def dot_repr(self):
        i, defs, links = self._dot_repr_recursive()
        out = """digraph G {{
            {0:s}
            {1:s}
        }}""".format("\n".join(defs), "\n".join(links))
        return out
        
    def _dot_repr_recursive(self, i=0):
        parent_i = i
        defs = ['A_{0:d} [label="{1:s}"];'.format(parent_i, self.label)]
        links = []
        i += 1
        for child in self.children:
            n, child_defs, child_links = child._dot_repr_recursive(i)
            defs.extend(child_defs)
            links.append('A_{0:d} -> A_{1:d};'.format(parent_i, i))
            i += n
        return (i, defs, links)


class ASTNode:
    pass



class IdentNode(ASTnode):
    pass


class NumberNode(ASTnode):
    pass


class ComputationNode(ASTnode):
    def __init__(self, vdecls, fdecls, stmts):
        self.vdecls = vdecls
        self.fdecls = fdecls
        self.stmts = stmts
        self.label = "computation"
        self.children = self.stmts


class VarDecl:
    def __init__(self, typ, ident):
        super().__init__()
        self.typ = typ
        self.ident = ident


class FuncDecl:
    pass


class AssignmentNode(ASTnode):
    def __init__(self):
        super().__init__()
        self.label = "assignment"


class IfNode(ASTnode):
    def __init__(self, children):
        super().__init__()
        self.label = "if"
        self.children = children


class WhileNode(ASTnode):
    def __init__(self, children):
        super().__init__()
        self.label = "while"
        self.children = children


class CallNode(ASTnode):
    def __init__(self):
        super().__init__()
        self.label = "call"


class ReturnNode(ASTnode):
    def __init__(self):
        super().__init__()
        self.label = "return"


class Computation(ASTnode):
    
    def __init__(self, vdecls, fdecls, stmts):
        super().__init__()
        self.vdecls = vdecls
        self.fdecls = fdecls
        self.stmts = stmts
        self.children = stmts
        self.label = "Computation"



