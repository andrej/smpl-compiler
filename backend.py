"""
Abstract base class for backend implementations.

The backend translates a stream of SSA IR instructions (inside a ssa.CompilationContext)
to machine-specific instructions. It can call back to a register allocator.

Author: André Rösti
"""

import ssa


class Backend:

    def __init__(self, ir):
        self.ir: ssa.CompilationContext = ir
        self.allocator = None
        self.block_instrs = {}  # Map block label to tuples (head_instrs, tail_instrs)
        self.instrs = []  # After linking: linear list of blocks
        self.block_offsets = {}  # Map block label to offset
        self.current_block = None

    def get_asm(self):
        inverse_block_offsets = {}
        for label, offset in self.block_offsets.items():
            if offset in inverse_block_offsets:
                inverse_block_offsets[offset].append(label)
            else:
                inverse_block_offsets[offset] = [label]

        out = ""
        label_length = max(len(label) for label in self.block_offsets)
        for offset, instr in enumerate(self.instrs):
            if offset in inverse_block_offsets:
                for label in inverse_block_offsets[offset]:
                    out += "{0:{1}s} ".format(label+":", label_length+1)
                out += str(instr)+"\n"
            else:
                out += "{}  {}\n".format(" "*label_length, str(instr))
        return out

    def get_machine_code(self):
        return b"".join(map(bytes, self.instrs))

    def compile(self):
        self.block_instrs = {}
        for block in self.ir:
            self.block_instrs[block.label] = ([], [], [])
        for block in self.ir:
            self.compile_block(block)

    def compile_prelude(self, func_block):
        pass

    def compile_epilogue(self, func_block):
        pass

    def compile_block(self, block):
        self.current_block = block.label
        if block.func.enter_block == block:
            self.compile_prelude(block)
        for instr in block.instrs:
            self.compile_instr(instr, context=block)
        if block.func.exit_block == block:
            self.compile_epilogue(block)

    def compile_instr(self, instr: ssa.Instruction, context: ssa.BasicBlock):
        raise NotImplementedError()

    def link(self):
        """
        Replace labels in jump instructions with actual addresses. This is only
        possible after code generation, when self.block_offsets is populated.
        :return:
        """
        self.instrs = []
        for block, (head_instrs, tail_instrs, term_instr) in self.block_instrs.items():
            self.block_offsets[block] = len(self.instrs)
            self.instrs.extend(head_instrs)
            self.instrs.extend(tail_instrs)
            self.instrs.extend(term_instr)

    def emit(self, instr, block=None):
        block = block or self.current_block
        self.block_instrs[block][0].append(instr)

    def emit_back(self, instr, block=None):
        block = block or self.current_block
        if not block in self.block_instrs:
            self.block_instrs[block] = ([], [], [])
        self.block_instrs[block][1].append(instr)

    def emit_term(self, instr, block=None):
        block = block or self.current_block
        self.block_instrs[block][2].append(instr)

    def emit_stack_load(self, offset, into, block=None, back=False):
        raise NotImplementedError()

    def emit_heap_load(self, offset, into, block=None, back=False):
        raise NotImplementedError()

    def emit_stack_store(self, offs, val_reg, block=None, back=False):
        raise NotImplementedError()
