"""
Abstract syntax tree nodes for the Smpl language. Provides methods that convert
AST nodes to SSA (single static assignment) instruction streams for compilation
and also methods for evaluation (interpreter).


Author: André Rösti
"""
import sys


class InterpreterContext:

    def __init__(self):
        self.functions = {}
        self.locals = {}


builtin_funcs = {
    "inputNum": lambda: int(input()),
    "outputNum": print,
    "outputNewLine": lambda: print("\n")
}


class ASTNode:

    def __init__(self, children=None, parent=None, val=None, label=None):
        if not children:
            children = []
        self.children = children 
        self.parent = parent
        self.label = label
        self.val = val

    def run(self, context: InterpreterContext):
        """
        Implements an interpreter for this AST. The context that is passed in must be updated
        according to the semantics of the language in subclasses. For expressions, the return
        value of the expression must be returned.
        :param context:
        :return: tuple (expression return value, new context)
        """
        sys.stderr.write("Warning: not implemented statement or expression skipped")
        return None, context

    def get_dot_edge_sets(self):
        return [DotEdgeSet(self.children, "", "black")]

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
        edge_sets = self.get_dot_edge_sets()
        for edge_set in edge_sets:
            for child in edge_set.children:
                n, child_defs, child_links = child._dot_repr_recursive(i)
                defs.extend(child_defs)
                links.append('A_{0:d} -> A_{1:d} [label="{2:s}", color={3:s}];'.format(parent_i, i, edge_set.label, edge_set.color))
                links.extend(child_links)
                i = n
        return i, defs, links


class ASTLeaf(ASTNode):

    def __init__(self, label):
        super().__init__()
        self.label = label


class DotEdgeSet:

    def __init__(self, children, label, color):
        self.children = children
        self.label = label
        self.color = color


class Identifier(ASTNode):

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.label = name

    def run(self, context):
        if self.name not in context.locals:
            raise Exception("Access to undeclared variable '{}'".format(self.name))
        val = context.locals[self.name]
        if val is None:
            sys.stderr.write("Warning: access to uninitialized variable '{}'".format(self.name))
            val = 0
        return val, context


class Number(ASTNode):

    def __init__(self, val):
        super().__init__()
        self.val = val
        self.label = "Immediate '{}'".format(val)

    def run(self, context):
        return self.val, context


class ArrayAccess(ASTNode):

    def __init__(self, identifier: Identifier, indices):
        super().__init__()
        self.identifier = identifier
        self.indices = indices
        self.label = "Array Access '{}{}'".format(identifier.name, "".join(["[{}]".format(i.label) for i in indices]))

    def run(self, context):
        name = self.identifier.name
        if name not in context.locals:
            raise Exception("Access to undeclared array '{}'".format(name))
        val = context.locals[name]
        for idx in self.indices:
            idx, context = idx.run(context)
            val = val[idx]
        if val is None:
            sys.stderr.write("Warning: Access to uninitialized array '{}'".format(name))
        return val, context


class VariableDeclaration(ASTNode):
    
    def __init__(self, ident: Identifier, dims=None):
        super().__init__()
        self.ident = ident
        self.dims = dims
        self.label = "Declare '{}'".format(ident.name)

    def run(self, context):
        name = self.ident.name
        if name in context.locals:
            raise Exception("Re-declaration of variable '{}'".format(name))
        if not self.dims:
            context.locals[name] = None  # uninitialized
        else:
            context.locals[name] = [None]
            todo = [context.locals[name]]
            for dim in self.dims:
                dim, context = dim.run(context)
                next_todo = []
                for arr in todo:
                    for i, _ in enumerate(arr):
                        arr[i] = [None] * dim
                        next_todo.append(arr[i])
                todo = next_todo
            context.locals[name] = context.locals[name][0]
        return None, context


class FunctionDeclaration(ASTNode):

    def __init__(self, ident, param_idents, body_vdecls, body_stmts, is_void=True):
        super().__init__()
        self.ident = ident
        self.param_idents = param_idents
        self.body_vdecls = body_vdecls
        self.body_stmts = body_stmts
        self.label = "Function '{}'".format(ident.label)

    def get_dot_edge_sets(self):
        return [DotEdgeSet(self.param_idents, "Param", "blue"),
                DotEdgeSet(self.body_vdecls, "Local Variable", "purple"),
                DotEdgeSet(self.body_stmts, "Body", "black")]

    def run(self, context):
        name = self.ident.name
        if name in context.functions:
            raise Exception("Re-declaration of function '{}'".format(name))
        context.functions[name] = self
        return None, context


class Computation(ASTNode):
    
    def __init__(self, vdecls, fdecls, stmts):
        super().__init__()
        self.vdecls = vdecls
        self.fdecls = fdecls
        self.stmts = stmts
        self.children = stmts
        self.label = "Computation"

    def get_dot_edge_sets(self):
        return [DotEdgeSet(self.stmts, "Statement", "black"),
                DotEdgeSet(self.vdecls, "Variable Declaration", "blue"),
                DotEdgeSet(self.fdecls, "Function Declaration", "blue")]

    def run(self, context=None):
        context = InterpreterContext()
        for vdecl in self.vdecls:
            _, context = vdecl.run(context)
        for fdecl in self.fdecls:
            _, context = fdecl.run(context)
        ret_val = None
        for stmt in self.stmts:
            ret_val, context = stmt.run(context)
            if ret_val is not None:
                break
        return ret_val, context


class BinOp(ASTNode):

    def __init__(self, op, opa, opb):
        super().__init__()
        assert op in {"+", "-", "*", "/", "<", "<=", ">", ">=", "==", "!="}
        self.op = op
        self.opa = opa
        self.opb = opb
        self.label = op
        self.children = [opa, opb]

    def run(self, context):
        lhs, context = self.opa.run(context)
        rhs, context = self.opb.run(context)
        res = None
        if self.op == "+":
            res = lhs + rhs
        elif self.op == "-":
            res = lhs - rhs
        elif self.op == "*":
            res = lhs * rhs
        elif self.op == "/":
            res = float(lhs) / rhs
        elif self.op == "<":
            res = lhs < rhs
        elif self.op == "<=":
            res = lhs <= rhs
        elif self.op == ">":
            res = lhs > rhs
        elif self.op == ">=":
            res = lhs >= rhs
        elif self.op == "==":
            res = lhs == rhs
        elif self.op == "!=":
            res = lhs != rhs
        return res, context


class Assignment(ASTNode):

    def __init__(self, lhs, rhs):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs
        self.label = "Assignment"

    def get_dot_edge_sets(self):
        return [DotEdgeSet([self.lhs], "LHS", "black"),
                DotEdgeSet([self.rhs], "RHS", "black")]

    def run(self, context):
        name = None
        if isinstance(self.lhs, Identifier):
            name = self.lhs.name
        elif isinstance(self.lhs, ArrayAccess):
            name = self.lhs.identifier.name
        if not name or name not in context.locals:
            raise Exception("Assignment to undeclared variable '{}'".format(name))
        val, context = self.rhs.run(context)
        if isinstance(self.lhs, Identifier):
            context.locals[name] = val
        else:
            arr = context.locals[name]
            for idx in self.lhs.indices[:-1]:
                arr = arr[idx]
            arr[self.lhs.indices[-1].val] = val
        return None, context


class FuncCall(ASTNode):

    def __init__(self, ident: Identifier, arg_exprs):
        super().__init__()
        self.ident = ident
        self.arg_exprs = arg_exprs
        self.label = "Call '{}'".format(ident.name)
        self.children = arg_exprs

    def run(self, context):
        fun_name = self.ident.name

        # Evaluate arguments
        args = []
        for arg_expr in self.arg_exprs:
            arg, context = arg_expr.run(context)
            args.append(arg)

        if fun_name in builtin_funcs:
            return builtin_funcs[fun_name](*args), context

        if fun_name not in context.functions:
            raise Exception("Call to undefined function '{}'".format(self.ident.name))
        fun: FunctionDeclaration = context.functions[fun_name]

        # Set up a new context for execution of the function body
        inner_context = InterpreterContext()
        inner_context.functions = context.functions
        for vdecl in fun.body_vdecls:
            _, inner_context = vdecl.run(inner_context)

        # Evaluate parameters and set them same as local variables
        # (parameters have pass-by-value semantics)
        for arg_ident, arg in zip(fun.param_idents, args):
            name = arg_ident.name
            if name in context.locals:
                raise Exception("Undefined behavior: '{}' is both a parameter and a local variable name".format(arg_name))
            inner_context.locals[name] = arg

        # Run the statements in the function body, terminating early if a return statement is encountered
        ret_val = None
        for stmt in fun.body_stmts:
            ret_val, inner_context = stmt.run(inner_context)
            if ret_val is not None:
                # Return statement executed; the following statements will be skipped
                break

        return ret_val, context


class IfStatement(ASTNode):

    def __init__(self, condition, stmts_if, stmts_else):
        super().__init__()
        self.condition = condition
        self.stmts_if = stmts_if
        self.stmts_else = stmts_else
        self.label = "if"

    def get_dot_edge_sets(self):
        return [DotEdgeSet([self.condition], "Condition", "black"),
                DotEdgeSet(self.stmts_if, "True", "green"),
                DotEdgeSet(self.stmts_else, "False", "red")]

    def dot_connected_nodes(self, typ=0):
        if typ == 0:
            return [self.condition]
        elif typ == 1:
            return self.stmts_if
        elif typ == 2:
            return self.stmts_else

    def run(self, context):
        if_eval, context = self.condition.run(context)
        stmts = self.stmts_if
        if not if_eval:
            stmts = self.stmts_else
        val = None
        for stmt in stmts:
            val, context = stmt.run(context)
        return val, context


class WhileStatement(ASTNode):

    def __init__(self, condition, body_stmts):
        super().__init__()
        self.condition = condition
        self.body_stmts = body_stmts
        self.label = "while"

    def get_dot_edge_sets(self):
        return [DotEdgeSet([self.condition], "Condition", "black"),
                DotEdgeSet(self.body_stmts, "Loop", "blue")]

    def run(self, context):
        val = None
        while True:
            condition_eval, context = self.condition.run(context)
            if not condition_eval:
                break
            for body_stmt in self.body_stmts:
                val, context = body_stmt.run(context)
        return val, context


class ReturnStatement(ASTNode):

    def __init__(self, value):
        super().__init__()
        self.value = value
        self.label = "return"
        self.children = [value]

    def run(self, context):
        return self.value.run(context)

