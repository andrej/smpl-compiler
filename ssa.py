"""
Class for representing single static assignment (SSA) instructions.
"""
import functools
import operator
import typing
import dot
import sys


class Op:
    pass


class ImmediateOp(Op):

    def __init__(self, val):
        self.val = val

    def __str__(self):
        return "#{}".format(self.val)

    def __eq__(self, other):
        return isinstance(other, ImmediateOp) and self.val == other.val


class UninitializedVarOpObj(Op):

    def __str__(self):
        return "??"


UninitializedVarOp = UninitializedVarOpObj()


class InstructionOp(Op):

    def __init__(self, instr):
        self.i = instr.i

    def __str__(self):
        return "({})".format(self.i)

    def __eq__(self, other):
        return isinstance(other, InstructionOp) and self.i == other.i


class ArgumentOp(Op):

    def __init__(self, ident):
        self.name = ident.name

    def __str__(self):
        return "@{}".format(self.name)

    def __eq__(self, other):
        return isinstance(other, ArgumentOp) and self.name == other.name


class LabelOp(Op):

    def __init__(self, label):
        self.label = label

    def __str__(self):
        return self.label

    def __eq__(self, other):
        return isinstance(other, LabelOp) and self.label == other.label


class FunctionOp(Op):

    def __init__(self, func):
        self.func = func

    def __str__(self):
        return self.func + "()"


class Instruction:

    def __init__(self, instr, *ops: Op, produces_output=True):
        self.instr: str = instr
        self.ops: typing.List[Op] = ops
        self.i: int = -1
        self.produces_output = produces_output
        self.dom_by_instr: typing.Optional[Instruction] = None  # linked list of previous dominating instructions

    def __str__(self):
        args = ", ".join(str(op) for op in self.ops)
        if not self.produces_output:
            return "{} {}".format(self.instr, args)
        return "{} = {} {}".format(self.i, self.instr, args)

    def dominance_class(self):
        """
        For purposes of common subexpression elimination, we consider certain instructions
        to be part of the same "dominance class":

        "adda" and "add" are equivalent for the purposes of CSE, but only in one direction
        (we can reuse the result of a add instruction instead of an "adda", but we
        probably do not want to use the result of an "adda", since then the compiler cannot
        group adda and load together into one instruction)

        "load" and "store" are part of the same dominance tree, since we have to consider
        stores to prevent aliasing of memory.
        :return:
        """
        if self.instr == "adda":
            return "add"
        if self.instr == "store":
            return "load"
        return self.instr

    def find_dominating_identical(self):
        """
        Find a dominating instruction that performs the identical computation. If we
        can find such an instruction, this instruction can be eliminated;
        1. The instruction is identical: It provides the value we require.
        2. The instruction dominates this one: The computed value is available in all control flow paths.

        :return:
        """
        dom_instr = self
        candidate = None
        while True:
            dom_instr = dom_instr.dom_by_instr  # not an infinite loop since domination is a tree; cannot be cyclic
            if not dom_instr:
                break
            if dom_instr.instr == "adda" and self.instr != "adda":
                # Cannot get the result of an "adda" instruction, because it will
                # most likely be merged into a three-operand combined load-and-add
                # by the backend
                instr = dom_instr
                continue
            if dom_instr.instr == "store":
                # A store kills a load, since the previous load might be invalidated
                # by this store (memory aliasing); but this is only the case if we can
                # prove that the loaded object is not identical
                break
            if self.ops == dom_instr.ops:
                candidate = dom_instr
                break
            instr = dom_instr
        return candidate  # none found


class Function(dot.DotSubgraph):

    def __init__(self):
        self.enter_block: BasicBlock = None
        self.exit_block: BasicBlock = None
        self.arg_names: typing.List = []
        self.is_main: bool = False
        self.name = ""

    def dot_roots(self):
        return [self.enter_block]

    def dot_label(self):
        return self.name


class BasicBlock(dot.DotNode):

    def __init__(self):
        super().__init__()
        self.instrs: typing.List[Instruction] = []
        self.label = ""
        self.succs: typing.List[BasicBlock] = []  # the main fall-through block should be the first in the list
        self.preds: typing.List[BasicBlock] = []
        self.locals_op: typing.Dict[str, Op] = {}
        self.locals_dim: typing.Dict[str, typing.List[int]] = {}
        self.func: Function = None
        self.dominates: typing.List[BasicBlock] = []  # list of basic blocks this block dominates
        self.dom_instr_tree: typing.Dict[str, Instruction] = {}  # search structure for common subexpression elimination

    def emit(self, instr_index, instr_name, *args, produces_output=True, may_eliminate=False):
        instr = Instruction(instr_name, *args, produces_output=produces_output)
        # Emitting an instruction in the same block means we need to update our search structure.
        # We know this will dominate the previous instruction because it is in the same block.
        dominance_class = instr.dominance_class()
        if dominance_class in self.dom_instr_tree:
            instr.dom_by_instr = self.dom_instr_tree[dominance_class]
        if may_eliminate:  # Common subexpression elimination
            identical = instr.find_dominating_identical()
            if identical:
                return InstructionOp(identical)
            elif (instr.instr == "load"
                  and len(self.instrs) > 0 and self.instrs[-1].instr == "adda"
                  and instr.ops[0] == InstructionOp(self.instrs[-1])):
                # Special handling of adda followed by load/store, which may only be
                # eliminated together or not at all, they must always appear in pairs.
                identical_adda = self.instrs[-1].find_dominating_identical()
                if identical_adda:
                    instr.ops = (InstructionOp(identical_adda),)
                    identical = instr.find_dominating_identical()
                    if identical:
                        # Have both adda and load that can be eliminated,
                        # thus eliminate both.
                        del self.instrs[-1]
                        return InstructionOp(identical)
        instr.i = instr_index
        self.instrs.append(instr)
        self.dom_instr_tree[dominance_class] = instr
        return InstructionOp(instr)

    def add_succ(self, block):
        self.succs.append(block)
        block.preds.append(self)

    def declare_local(self, name: str, dims: typing.Optional[typing.List[int]]):
        if name in self.locals_op:
            raise Exception("Attempting to redeclare local '{}'".format(name))
        self.locals_op[name] = None
        self.locals_dim[name] = dims

    def get_local(self, name):
        if name not in self.locals_op:
            raise Exception("Access to undeclared local '{}'".format(name))
        val_op = self.locals_op[name]
        dims = self.locals_dim[name]
        if not val_op:
            sys.stderr.write("Warning: access to uninitialized variable '{}'\n".format(name))
            return UninitializedVarOp, dims
        return val_op, dims

    def set_local_op(self, name, val: Op):
        if name not in self.locals_op:
            raise Exception("Access to undeclared local '{}'".format(name))
        self.locals_op[name] = val

    def rename_op(self, old_op, new_op, visited=None):
        """
        Recursively rename an operator appearing in all instructions in this block
        and all of its successors.
        """
        if not visited:
            visited = set()
        if self in visited:
            return
        for instr in self.instrs:
            new_instr_ops = list(instr.ops)
            for i, op in enumerate(instr.ops):
                if op == old_op:
                    new_instr_ops[i] = new_op
            instr.ops = tuple(new_instr_ops)
        for succ in self.succs:
            visited.add(succ)
            succ.rename_op(old_op, new_op, visited)

    def dot_edge_sets(self):
        never_falls_through = self.instrs and self.instrs[-1].instr in {"bra", "ret"}
        edge_sets = []
        if not never_falls_through:
            edge_sets += [dot.DotEdgeSet(self.succs[0:1], label="fall-through", color="black"),
                          dot.DotEdgeSet(self.succs[1:], label="branch", color="black")]
        else:
            edge_sets += [dot.DotEdgeSet(self.succs, label="jump", color="black")]
        edge_sets += [dot.DotEdgeSet(self.dominates, label="dom", color="blue", style="dotted")]
        return edge_sets

    def dot_label(self):
        instr_reps = map(str, self.instrs)
        out = "<b>{0:s} | {{ {1:s} }}".format(self.label, " | ".join(instr_reps))
        return out


class CompilationContext(dot.DotGraph):
    """
    The compilation context is the root element of the intermediate representation of the program.
    It contains references to all the basic blocks. During compilation, a pointer to the
    "current_block" is updated; all emitted instructions are appended to this current block. As
    compilation progresses, further basic blocks can split of the current block using the
    get_new_block_with_same_context() function; this freezes the current block's context
    (mainly variable mappings). For completely independent blocks (new functions), use the
    get_new_block() function.
    """

    def __init__(self, do_cse=True):
        super().__init__()
        self.instr_counter = 1  # Start at 1 for consistency with assignment
        self.block_counter = 0
        self.root_blocks = []
        self.current_block: typing.Optional[BasicBlock] = None
        self.do_cse = do_cse

    def __iter__(self):
        return CompilationContextIterator(self)

    def emit(self, *args, produces_output=True, may_eliminate=False):
        """
        Emit a new IR SSA instruction in the currently active block.
        :param args:
        :param produces_output: Whether this produces a value or not (for visualization only)
        :param may_eliminate: Whether common subexpression elimination may be performed for this instruction or not
        :return:
        """
        if not self.do_cse:
            may_eliminate = False
        result_op = self.current_block.emit(self.instr_counter, *args, produces_output=produces_output,
                                            may_eliminate=may_eliminate)
        self.instr_counter += 1
        return result_op

    def get_new_block(self):
        block = BasicBlock()
        self.block_counter += 1
        block.label = "BB{}".format(self.block_counter)
        return block

    def get_new_block_with_same_context(self):
        """
        Return an empty basic block with a copy of the current blocks context
        (local variables). This is useful for creating successor blocks that
        may want to access locals from the predecessor block.
        :return:
        """
        block = self.get_new_block()
        block.locals_op = self.current_block.locals_op.copy()
        block.locals_dim = self.current_block.locals_dim.copy()
        block.dom_instr_tree = self.current_block.dom_instr_tree.copy()
        block.func = self.current_block.func
        return block

    def set_current_block(self, block: BasicBlock):
        self.current_block = block

    def add_root_block(self, block: BasicBlock):
        self.root_blocks.append(block)

    def eliminate_constants(self):
        """
        Perform constant elimination by applying the following two optimizations:
        - Instructions on two immediates: Compute at compile time
        - Instructions on one value and one immediate: If the immediate is
          the idenity of that operation, eliminate the instruction
        """
        pyfuns = {"add": operator.add,
                  "adda": operator.add,
                  "sub": operator.sub,
                  "mul": operator.mul,
                  "cmp": lambda a, b: 0 if a == b else -1 if a < b else +1}
        left_identities = {"add": ImmediateOp(0),
                           "adda": ImmediateOp(0),
                           "mul": ImmediateOp(1)}
        right_identities = {"add": ImmediateOp(0),
                            "adda": ImmediateOp(0),
                            "sub": ImmediateOp(0),
                            "mul": ImmediateOp(1),
                            "div": ImmediateOp(1)}
        n_eliminated = 1
        while n_eliminated > 0:  # need to keep iterating until we reach a fixed point
            n_eliminated = 0
            for block in self:
                for i, instr in enumerate(block.instrs):
                    iname = instr.instr
                    if iname in pyfuns:
                        assert len(instr.ops) == 2
                        if isinstance(instr.ops[0], ImmediateOp) and isinstance(instr.ops[1], ImmediateOp):
                            res = pyfuns[instr.instr](instr.ops[0].val, instr.ops[1].val)
                            del block.instrs[i]
                            block.rename_op(InstructionOp(instr), ImmediateOp(res))
                            n_eliminated += 1
                        elif iname in left_identities and instr.ops[0] == left_identities[iname]:
                            del block.instrs[i]
                            block.rename_op(InstructionOp(instr), instr.ops[1])
                            n_eliminated += 1
                        elif iname in right_identities and instr.ops[1] == right_identities[iname]:
                            del block.instrs[i]
                            block.rename_op(InstructionOp(instr), instr.ops[0])
                            n_eliminated += 1

    def eliminate_dead_code(self):
        """
        Remove instructions from the code if their values are never used, and
        if they have no side effects.
        """
        used_set = {op.i
                    for block in self
                    for instr in block.instrs
                    for op in instr.ops
                    if isinstance(op, InstructionOp)}
        for block in self:
            for i, instr in enumerate(block.instrs):
                if instr.produces_output and instr.i not in used_set:
                    del block.instrs[i]

    def dot_subgraphs(self):
        funcs = {bb.func for bb in self.root_blocks}
        return list(funcs)


class CompilationContextIterator:
    """
    Depth-first iteration gives the correct order of blocks for fall-through execution.
    """

    def __init__(self, context: CompilationContext):
        self.context = context
        self.todo = self.context.root_blocks.copy()
        self.visited = set()

    def __next__(self):
        if not self.todo:
            raise StopIteration
        block = self.todo.pop(0)
        self.visited.add(block)
        for succ in block.succs:
            if succ in self.visited:
                continue
            self.visited.add(succ)
            self.todo.append(succ)
        return block