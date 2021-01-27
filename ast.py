"""
Abstract syntax tree nodes for the Smpl language. Provides methods that convert
AST nodes to SSA (single static assignment) instruction streams for compilation
and also methods for evaluation (interpreter).


Author: André Rösti
"""
import sys
import functools
import operator
import ssa
import config
import dot


class InterpreterContext:

    def __init__(self):
        self.functions = {}
        self.locals = {}


builtin_funcs = {
    "inputNum": lambda: int(input()),
    "outputNum": print,
    "outputNewLine": lambda: print("\n")
}


class ASTNode(dot.DotGraph, dot.DotNode):

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

    def compile(self, context: ssa.CompilationContext):
        """
        Compile the given statement or expression to SSA IR by appropriately adding instructions
        to the given compilation context and updating the context's local variable table. After
        compilation of a statement, the context should be updated appropriately such that
        following statements can be added by updating the same context.
        :param context:
        :return: For expressions, return an ssa.Op representing the result of the expression
        """
        raise NotImplementedError()

    def dot_roots(self):
        return [self]


class Identifier(ASTNode):

    def __init__(self, name):
        super().__init__()
        self.name = name

    def run(self, context):
        if self.name not in context.locals:
            raise Exception("Access to undeclared variable '{}'".format(self.name))
        val = context.locals[self.name]
        if val is None:
            sys.stderr.write("Warning: access to uninitialized variable '{}'\n".format(self.name))
            val = 0
        return val, context

    def compile(self, context):
        val_op, _ = context.current_block.get_local(self.name)
        if not val_op:
            sys.stderr.write("Warning: access to uninitialized variable '{}'\n".format(self.name))
            val_op = ssa.ImmediateOp(0)
        return val_op

    def dot_label(self):
        return self.name


class Number(ASTNode):

    def __init__(self, val):
        super().__init__()
        self.val = val

    def run(self, context):
        return self.val, context

    def compile(self, context):
        return ssa.ImmediateOp(self.val)

    def dot_label(self):
        return self.val


class ArrayAccess(ASTNode):

    def __init__(self, identifier: Identifier, indices):
        super().__init__()
        self.identifier = identifier
        self.indices = indices

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

    def compile(self, context):
        addr_op = self.compile_addr(context)
        load_op = context.emit("load", addr_op)
        return load_op

    def compile_addr(self, context):
        name = self.identifier.name
        base_addr_op, dims = context.current_block.get_local(name)
        offset_op = ssa.ImmediateOp(0)
        for i, idx in enumerate(self.indices):
            idx_op = idx.compile(context)
            stride = functools.reduce(operator.mul, dims[i+1:], 1)
            this_offset_op = context.emit("mul", idx_op, ssa.ImmediateOp(stride))
            offset_op = context.emit("add", offset_op, this_offset_op)
        offset_op = context.emit("mul", offset_op, ssa.ImmediateOp(config.INTEGER_SIZE))
        addr_op = context.emit("adda", base_addr_op, offset_op)
        return addr_op

    def dot_label(self):
        return "Array Access '{}{}'".format(self.identifier.name, "".join(["[{}]".format(i.dot_label()) for i in self.indices]))


class VariableDeclaration(ASTNode):
    
    def __init__(self, ident: Identifier, dims=None):
        super().__init__()
        self.ident = ident
        self.dims = dims

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

    def compile(self, context):
        name = self.ident.name
        dims = self.dims
        if dims:
            dims = [dim.val for dim in dims]
        context.current_block.declare_local(name, dims)
        if dims:  # issue allocation instruction
            size = functools.reduce(operator.mul, dims, 1) * config.INTEGER_SIZE
            base_addr = context.emit("alloca", size)
            context.current_block.set_local_op(name, base_addr)
        return None

    def dot_label(self):
        return "Declare '{}'".format(self.ident.name)


class FunctionDeclaration(ASTNode):

    def __init__(self, ident, param_idents, body_vdecls, body_stmts, is_void=True):
        super().__init__()
        self.ident = ident
        self.param_idents = param_idents
        self.body_vdecls = body_vdecls
        self.body_stmts = body_stmts

    def dot_edge_sets(self):
        return [dot.DotEdgeSet(self.param_idents, "Param", "blue"),
                dot.DotEdgeSet(self.body_vdecls, "Local Variable", "purple"),
                dot.DotEdgeSet(self.body_stmts, "Body", "black")]

    def run(self, context):
        name = self.ident.name
        if name in context.functions:
            raise Exception("Re-declaration of function '{}'".format(name))
        context.functions[name] = self
        return None, context

    def compile(self, context):
        old_root = context.current_block
        root = context.get_new_block()
        context.add_root_block(root)
        context.set_current_block(root)

        # add function arguments
        for param_ident in self.param_idents:
            context.current_block.declare_local(param_ident.name, None)
            context.current_block.set_local_op(param_ident.name, ssa.ArgumentOp(param_ident))

        # add locals
        for vdecl in self.body_vdecls:
            vdecl.compile(context)

        # Compile body
        for stmt in self.body_stmts:
            stmt.compile(context)

        context.set_current_block(old_root)

    def dot_label(self):
        return "Function '{}'".format(self.ident.dot_label())


class Computation(ASTNode):
    
    def __init__(self, vdecls, fdecls, stmts):
        super().__init__()
        self.vdecls = vdecls
        self.fdecls = fdecls
        self.stmts = stmts

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

    def compile(self, context):
        root_block = context.get_new_block()
        context.add_root_block(root_block)
        context.set_current_block(root_block)
        for vdecl in self.vdecls:
            vdecl.compile(context)
        for fdecl in self.fdecls:
            fdecl.compile(context)
        for stmt in self.stmts:
            stmt.compile(context)
        context.emit("end")
        return None

    def dot_label(self):
        return "Computation"

    def dot_edge_sets(self):
        return [dot.DotEdgeSet(self.stmts, "Statement", "black"),
                dot.DotEdgeSet(self.vdecls, "Variable Declaration", "blue"),
                dot.DotEdgeSet(self.fdecls, "Function Declaration", "blue")]


class BinOp(ASTNode):

    def __init__(self, op, opa, opb):
        super().__init__()
        assert op in {"+", "-", "*", "/", "<", "<=", ">", ">=", "==", "!="}
        self.op = op
        self.opa = opa
        self.opb = opb

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

    def compile(self, context):
        instr_map = {"+": "add",
                     "-": "sub",
                     "*": "mul",
                     "/": "div",
                     "<": "cmp", "<=": "cmp", ">": "cmp", ">=": "cmp", "==": "cmp", "!=": "cmp"}
        op_a = self.opa.compile(context)
        op_b = self.opb.compile(context)
        res_op = context.emit(instr_map[self.op], op_a, op_b)
        return res_op

    def compile_conditional_jump(self, context, jump_label):
        """
        Compiles a conditional jump that is performed on the *opposite* of the condition,
        i.e. if the condition holds true, execution falls through, but if it is false,
        a jump is performed. This is more usful for while/if code generation.
        :param context:
        :param jump_label:
        :return:
        """
        cond_op = self.compile(context)
        branch_map = {">=": "blt", ">": "ble", "<=": "bgt", "<": "bge", "!=": "beq", "==": "bne"}
        context.emit(branch_map[self.op], cond_op, jump_label)
        return None

    def dot_label(self):
        return self.op

    def dot_edge_sets(self):
        return [dot.DotEdgeSet([self.opa], "Operand A", "black"),
                dot.DotEdgeSet([self.opb], "Operand B", "black")]


class Assignment(ASTNode):

    def __init__(self, lhs, rhs):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def dot_edge_sets(self):
        return [dot.DotEdgeSet([self.lhs], "LHS", "black"),
                dot.DotEdgeSet([self.rhs], "RHS", "black")]

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

    def compile(self, context):
        val_op = self.rhs.compile(context)
        if isinstance(self.lhs, Identifier):
            context.current_block.set_local_op(self.lhs.name, val_op)
            # Emit no additional instructions; simply update local context s.t. it refers to new value

        elif isinstance(self.lhs, ArrayAccess):
            name = self.lhs.identifier.name
            if name not in context.current_block.locals_op:
                raise Exception("Assignment to undeclared array '{}'".format(name))

            addr_op = self.lhs.compile_addr(context)
            context.emit("store", val_op, addr_op)

        return None

    def dot_label(self):
        return "Assignment"


class FuncCall(ASTNode):

    def __init__(self, ident: Identifier, arg_exprs):
        super().__init__()
        self.ident = ident
        self.arg_exprs = arg_exprs

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
                raise Exception("Undefined behavior: '{}' is both a parameter and a local variable name".format(name))
            inner_context.locals[name] = arg

        # Run the statements in the function body, terminating early if a return statement is encountered
        ret_val = None
        for stmt in fun.body_stmts:
            ret_val, inner_context = stmt.run(inner_context)
            if ret_val is not None:
                # Return statement executed; the following statements will be skipped
                break

        return ret_val, context

    def compile(self, context):
        arg_ops = []
        for arg_expr in self.arg_exprs:
            arg_op = arg_expr.compile(context)
            arg_ops.append(arg_op)
        res_op = context.emit("call", *arg_ops)
        return res_op

    def dot_label(self):
        return "Call '{}'".format(self.ident.name)

    def dot_edge_sets(self):
        return [dot.DotEdgeSet(self.arg_exprs, "Argument", "black")]


class IfStatement(ASTNode):

    def __init__(self, condition, stmts_if, stmts_else):
        super().__init__()
        self.condition = condition
        self.stmts_if = stmts_if
        self.stmts_else = stmts_else

    def dot_edge_sets(self):
        return [dot.DotEdgeSet([self.condition], "Condition", "black"),
                dot.DotEdgeSet(self.stmts_if, "True", "green"),
                dot.DotEdgeSet(self.stmts_else, "False", "red")]

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

    def compile(self, context):
        then_block = context.get_new_block_with_same_context()
        else_block = context.get_new_block_with_same_context()
        join_block = context.get_new_block_with_same_context()
        context.current_block.add_succ(then_block)
        context.current_block.add_succ(else_block)

        # Compile condition evaluation.
        self.condition.compile_conditional_jump(context, else_block.label)
        # fall through if condition holds true

        # Compile both branches.
        context.set_current_block(then_block)
        for stmt in self.stmts_if:
            stmt.compile(context)
        context.emit("bra", join_block.label)  # skip over the else block
        # note that after compiling all these statements, current_block is not necessarily then_block
        context.current_block.add_succ(join_block)
        then_block = context.current_block

        context.set_current_block(else_block)
        for stmt in self.stmts_else:
            stmt.compile(context)
        # current_block is not necessarily else_block
        context.current_block.add_succ(join_block)
        else_block = context.current_block
        # fall-through into join block

        # In join block, add phi nodes for locals that have been touched by either branch.
        # Since variables must be declared ahead of time,
        context.set_current_block(join_block)
        for name in context.current_block.locals_op:
            val_a, dims = then_block.get_local(name)
            val_b, _ = else_block.get_local(name)
            if val_a == val_b:
                continue
            phi_op = context.emit("phi", val_a, val_b)
            context.current_block.set_local_op(name, phi_op)

        # current context has been updated to join block, so any future instructions will
        # be added to end of it
        return None

    def dot_label(self):
        return "if"


class WhileStatement(ASTNode):

    def __init__(self, condition, body_stmts):
        super().__init__()
        self.condition = condition
        self.body_stmts = body_stmts

    def dot_edge_sets(self):
        return [dot.DotEdgeSet([self.condition], "Condition", "black"),
                dot.DotEdgeSet(self.body_stmts, "Loop", "blue")]

    def run(self, context):
        val = None
        while True:
            condition_eval, context = self.condition.run(context)
            if not condition_eval:
                break
            for body_stmt in self.body_stmts:
                val, context = body_stmt.run(context)
        return val, context

    def compile(self, context):

        join_block = context.get_new_block_with_same_context()
        context.current_block.add_succ(join_block)

        # We make a temporary context and compile all the body statements.
        # This will all be discarded; we only use it to see which variables
        # are assigned to, so we can add the correct phi nodes. We then re-
        # compile the body using the phi node values of the touched variables
        # after the join block.
        old_instr_counter = context.instr_counter
        tmp_body_block = context.get_new_block_with_same_context()
        context.set_current_block(tmp_body_block)
        for stmt in self.body_stmts:
            stmt.compile(context)
        tmp_body_block = context.current_block

        context.set_current_block(join_block)
        for name in context.current_block.locals_op:
            val_a, _ = context.current_block.get_local(name)
            val_b, _ = tmp_body_block.get_local(name)
            if val_a != val_b:
                phi_op = context.emit("phi", val_a, val_b)
                context.current_block.set_local_op(name, phi_op)

        # Compile condition (after phi node)
        exit_block = context.get_new_block_with_same_context()
        self.condition.compile_conditional_jump(context, exit_block.label)
        # fall through into loop body
        new_instr_counter = context.instr_counter

        body_block = context.get_new_block_with_same_context()
        context.instr_counter = old_instr_counter  # reset the instruction counter because we are redoing the body instructions
        context.set_current_block(body_block)
        for stmt in self.body_stmts:
            stmt.compile(context)
        # note that after compiling these statements, context.current_block is not necessarily equal to body_block
        context.instr_counter = new_instr_counter  # reset to before the reset
        context.emit("bra", join_block.label)

        join_block.add_succ(body_block)  # fall-through
        context.current_block.add_succ(join_block)  # loop
        join_block.add_succ(exit_block)  # branch-out

        context.set_current_block(exit_block)
        return None

    def dot_label(self):
        return "while"


class ReturnStatement(ASTNode):

    def __init__(self, value):
        super().__init__()
        self.value = value

    def run(self, context):
        return self.value.run(context)

    def compile(self, context):
        res_op = self.value.compile(context)
        context.emit("return", res_op)
        return res_op

    def dot_label(self):
        return "return"

    def dot_edge_sets(self):
        return [dot.DotEdgeSet([self.value], "return value", "black")]

