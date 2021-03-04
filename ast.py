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


class AST(dot.DotSubgraph):

    def __init__(self, root):
        self.root = root

    def compile(self, *args):
        return self.root.compile(*args)

    def run(self):
        return self.root.run()

    def dot_subgraphs(self):
        return [self]

    def dot_roots(self):
        return [self.root]


class ASTNode(dot.DotNode):

    def run(self, context):
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
        load_op = context.emit("load", addr_op, may_eliminate=True)
        # We must check here whether we can eliminate BOTH the load and
        # the adda instruction; cannot just eliminate one of both, since
        # they are supposed to appear in pairs.
        return load_op

    def compile_addr(self, context):
        name = self.identifier.name
        base_addr_op, dims = context.current_block.get_local(name)
        offset_op = ssa.ImmediateOp(0)
        for i, idx in enumerate(self.indices):
            idx_op = idx.compile(context)
            stride = functools.reduce(operator.mul, dims[i+1:], 1)
            this_offset_op = context.emit("mul", idx_op, ssa.ImmediateOp(stride), may_eliminate=True)
            offset_op = context.emit("add", offset_op, this_offset_op, may_eliminate=True)
        offset_op = context.emit("mul", offset_op, ssa.ImmediateOp(config.INTEGER_SIZE), may_eliminate=True)
        addr_op = context.emit("adda", base_addr_op, offset_op, may_eliminate=False)
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
            base_addr = context.emit("alloca", ssa.ImmediateOp(size))
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
        return [dot.DotEdgeSet(self.param_idents, label="Param"),
                dot.DotEdgeSet(self.body_vdecls, label="Local Variable", color="purple"),
                dot.DotEdgeSet(self.body_stmts, label="Body")]

    def run(self, context):
        name = self.ident.name
        if name in context.functions:
            raise Exception("Re-declaration of function '{}'".format(name))
        context.functions[name] = self
        return None, context

    def compile(self, context):
        old_root = context.current_block
        root = context.get_new_block()
        func = ssa.Function()
        func.name = self.ident.name
        func.enter_block = root
        func.arg_names = [ident.name for ident in self.param_idents]
        root.func = func
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

        # Invariant that must hold here: compiling statements results in one join block in the end
        func.exit_block = context.current_block

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
        main_func = ssa.Function()
        main_func.name = "main"
        main_func.is_main = True
        main_func.enter_block = root_block
        root_block.func = main_func
        context.add_root_block(root_block)
        context.set_current_block(root_block)
        for vdecl in self.vdecls:
            vdecl.compile(context)
        for fdecl in self.fdecls:
            fdecl.compile(context)
        for stmt in self.stmts:
            stmt.compile(context)
        main_func.exit_block = main_func
        context.emit("end", produces_output=False)
        return None

    def dot_label(self):
        return "Computation"

    def dot_edge_sets(self):
        return [dot.DotEdgeSet(self.stmts, label="Statement"),
                dot.DotEdgeSet(self.vdecls, label="Variable Declaration", color="blue"),
                dot.DotEdgeSet(self.fdecls, label="Function Declaration", color="blue")]


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
        res_op = context.emit(instr_map[self.op], op_a, op_b, may_eliminate=True)
        return res_op

    def compile_conditional_jump(self, context, jump_block):
        """
        Compiles a conditional jump that is performed on the *opposite* of the condition,
        i.e. if the condition holds true, execution falls through, but if it is false,
        a jump is performed. This is more usful for while/if code generation.
        :param context:
        :param jump_block:
        :return:
        """
        cond_op = self.compile(context)
        branch_map = {">=": "blt", ">": "ble", "<=": "bgt", "<": "bge", "!=": "beq", "==": "bne"}
        context.emit(branch_map[self.op], cond_op, ssa.LabelOp(jump_block), produces_output=False)
        return None

    def dot_label(self):
        return dot.label_escape(self.op)

    def dot_edge_sets(self):
        return [dot.DotEdgeSet([self.opa], label="Operand A"),
                dot.DotEdgeSet([self.opb], label="Operand B")]


class Assignment(ASTNode):

    def __init__(self, lhs, rhs):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def dot_edge_sets(self):
        return [dot.DotEdgeSet([self.lhs], label="LHS"),
                dot.DotEdgeSet([self.rhs], label="RHS")]

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
                idx, context = idx.run(context)
                arr = arr[idx]
            last_index, context = self.lhs.indices[-1].run(context)
            arr[last_index] = val
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
            context.emit("store", val_op, addr_op, produces_output=False)

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
        res_op = context.emit("call", ssa.FunctionOp(self.ident.name), *arg_ops,
                              may_eliminate=False)
        return res_op

    def dot_label(self):
        return "Call '{}'".format(self.ident.name)

    def dot_edge_sets(self):
        return [dot.DotEdgeSet(self.arg_exprs, label="Argument")]


class IfStatement(ASTNode):

    def __init__(self, condition, stmts_if, stmts_else):
        super().__init__()
        self.condition = condition
        self.stmts_if = stmts_if
        self.stmts_else = stmts_else

    def dot_edge_sets(self):
        return [dot.DotEdgeSet([self.condition], label="Condition"),
                dot.DotEdgeSet(self.stmts_if, label="True", color="green"),
                dot.DotEdgeSet(self.stmts_else, label="False", color="red")]

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

        # We know dominator information for if structures
        context.current_block.dominates.extend([then_block, else_block, join_block])

        # Compile condition evaluation.
        self.condition.compile_conditional_jump(context, else_block)
        # fall through if condition holds true

        # Compile both branches.
        context.set_current_block(then_block)
        for stmt in self.stmts_if:
            stmt.compile(context)
        context.emit("bra", ssa.LabelOp(join_block), produces_output=False)  # skip over the else block
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
            phi_op = context.emit("phi",
                                  ssa.LabelOp(then_block), val_a,
                                  ssa.LabelOp(else_block), val_b)
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
        return [dot.DotEdgeSet([self.condition], label="Condition"),
                dot.DotEdgeSet(self.body_stmts, label="Loop", color="blue")]

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

        original_block = context.current_block
        head_block = context.get_new_block_with_same_context()
        # Fall-through into the loop header, which contains the condition evaluation.
        # We also jump to this block again at the end of the loop to re-evaluate.
        # After compiling the loop body (and seeing which variables it uses), we
        # will also add the necessary phi nodes here and rename the variables in the loop body.
        context.current_block.dominates.append(head_block)
        context.current_block.add_succ(head_block)

        # Loop body
        body_block = context.get_new_block_with_same_context()
        for name, op in body_block.locals_op.items():
            # We mark all operands as "possibly phi"; this helps us observe
            # which operands are touched by the body instructions later.
            # Note that for nested loops, this means operands will be nested
            # in multiple layers of PossiblyPhiOps.
            body_block.locals_op[name] = ssa.PossiblyPhiOp(op)
        context.set_current_block(body_block)
        for stmt in self.body_stmts:
            stmt.compile(context)
        # note that after compiling these statements, context.current_block is not necessarily
        # equal to body_block any more! However, since all of our control has one singular
        # join block, we know that all control in the loop body ends in context.current_block.
        context.emit("bra", ssa.LabelOp(head_block), produces_output=False)
        body_end_block = context.current_block

        # Emit necessary phi nodes; look at all the variables assigned to in the loop body.
        context.set_current_block(head_block)  # go back to head block to emit phis
        for name in body_end_block.locals_op:
            head_op, _ = head_block.get_local(name)
            wrapped_head_op = ssa.PossiblyPhiOp(head_op)
            body_op, _ = body_end_block.get_local(name)
            if body_op != wrapped_head_op:  # The mapping name -> op changed in the child block!
                renamed_op = context.emit("phi",
                                          ssa.LabelOp(original_block), head_op,
                                          ssa.LabelOp(body_end_block), body_op)
                body_block.rename_op(wrapped_head_op, renamed_op)
                head_block.set_local_op(name, renamed_op)
                body_block.set_local_op(name, renamed_op)
            else:  # Operand not touched; undo wrapping
                body_block.rename_op(wrapped_head_op, head_op)

        # Compile condition (after phi nodes)
        exit_block = context.get_new_block_with_same_context()
        self.condition.compile_conditional_jump(context, exit_block)

        head_block.add_succ(body_block)  # fall-through
        head_block.add_succ(exit_block)  # branch-out
        body_end_block.add_succ(head_block)  # loop

        # Set dominator information known ahead of time for this control structure
        head_block.dominates.extend([body_block, exit_block])

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
        # FIXME hoist into join block for if statements!
        res_op = self.value.compile(context)
        context.emit("return", res_op, produces_output=False)
        return res_op

    def dot_label(self):
        return "return"

    def dot_edge_sets(self):
        return [dot.DotEdgeSet([self.value] if self.value else [], label="return value", color="black")]


class InterpreterContext:

    def __init__(self):
        self.functions = {}
        self.locals = {}


builtin_funcs = {
    "inputNum": lambda: int(input()),
    "outputNum": print,
    "outputNewLine": lambda: print("\n")
}


