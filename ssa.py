"""
Class for representing single static assignment (SSA) instructions.
"""
import typing
import dot


class Op:
    pass


class ImmediateOp(Op):
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return "#{}".format(self.val)


class InstructionOp(Op):
    def __init__(self, instr):
        self.i = instr.i

    def __str__(self):
        return "({})".format(self.i)


class ArgumentOp(Op):
    def __init__(self, ident):
        self.name = ident.name

    def __str__(self):
        return "@{}".format(self.name)


class LabelOp(Op):
    def __init__(self, block):
        self.label = block.label

    def __str__(self):
        return self.label


class Instruction:

    def __init__(self, instr, *ops: Op):
        self.instr: str = instr
        self.ops: typing.Iterable[Op] = ops
        self.i: int = -1

    def __str__(self):
        return "{} = {} {}".format(self.i, self.instr, ", ".join(str(op) for op in self.ops))


class BasicBlock(dot.DotNode):

    def __init__(self):
        super().__init__()
        self.instrs = []
        self.label = ""
        self.succs: typing.List[BasicBlock] = [] # the main fall-through block should be the first in the list
        self.locals_op: typing.Dict[str, Op] = {}
        self.locals_dim: typing.Dict[str, typing.List[int]] = {}

    def emit(self, instr_index, *args):
        instr = Instruction(*args)
        instr.i = instr_index
        self.instrs.append(instr)
        return InstructionOp(instr)

    def add_succ(self, block):
        self.succs.append(block)

    def declare_local(self, name: str, dims: typing.Optional[typing.List[int]]):
        if name in self.locals_op:
            raise Exception("Attempting to redeclare local '{}'".format(name))
        self.locals_op[name] = None
        self.locals_dim[name] = dims

    def get_local(self, name):
        if name not in self.locals_op:
            raise Exception("Access to undeclared local '{}'".format(name))
        return self.locals_op[name], self.locals_dim[name]

    def set_local_op(self, name, val: Op):
        if name not in self.locals_op:
            raise Exception("Access to undeclared local '{}'".format(name))
        self.locals_op[name] = val

    def dot_edge_sets(self):
        return [dot.DotEdgeSet(self.succs[0:1], "next", "black"),
                dot.DotEdgeSet(self.succs[1:], "jump", "black")]

    def dot_label(self):
        instr_reps = map(str, self.instrs)
        out = "<b>{0:s} | {{ {1:s} }}".format(self.label, " | ".join(instr_reps))
        return out


class CompilationContext(dot.DotGraph):

    def __init__(self):
        super().__init__()
        self.instr_counter = 0
        self.block_counter = 0
        self.root_blocks = []
        self.current_block: BasicBlock = None

    def emit(self, *args):
        result_op = self.current_block.emit(self.instr_counter, *args)
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
        return block

    def set_current_block(self, block: BasicBlock):
        self.current_block = block

    def add_root_block(self, block: BasicBlock):
        self.root_blocks.append(block)

    def dot_roots(self):
        return self.root_blocks

