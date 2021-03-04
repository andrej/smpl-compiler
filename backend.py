"""
Abstract base class for backend implementations.

The backend translates a stream of SSA IR instructions (inside a ssa.CompilationContext)
to machine-specific instructions. It can call back to a register allocator.

Author: André Rösti
"""

import ssa
import math


class Backend:

    def __init__(self, ir):
        self.ir: ssa.CompilationContext = ir
        self.allocator = None
        self.block_instrs = {}  # Map block label to tuples (head_instrs, tail_instrs)
        self.instrs = []  # After linking: linear list of blocks
        self.block_offsets = {}  # Map block label to offset
        self.func_entry_offsets = {}  # Map function name to its prologue
        self.func_exit_offsets = {}  # Map function name to its epilogue
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
        num_length = math.floor(math.log10(len(self.instrs))) + 1
        for offset, instr in enumerate(self.instrs):
            out += "{0:{1}d}  ".format(offset, num_length)
            if offset in inverse_block_offsets:
                for label in inverse_block_offsets[offset]:
                    out += "{0:{1}s} ".format(label+":", label_length+1)
                out += str(instr)+"\n"
            else:
                out += "{}  {}\n".format(" "*(label_length), str(instr))
        return out

    def get_machine_code(self):
        return b"".join(map(bytes, self.instrs))

    def compile(self):
        # New "program loader" block for init code
        self.block_instrs = {}
        for block in self.ir:
            self.block_instrs[block.label] = ([], [], [])
        self.current_block = self.ir.root_blocks[0].label  # Main block
        self.compile_init()
        for block in self.ir:
            self.compile_block(block)

    def compile_init(self):
        """
        "Loader" code compiled into the program. This only gets executed once at
        the start of the program before the main function.
        """
        pass

    def compile_prologue(self, func_block):
        """
        Function prologue. Save callee-save registers here.
        """
        pass

    def compile_epilogue(self, func_block):
        """
        Function epilogue. All function exits should jump to the stream of code
        compiled here (its address offset is stored in self.func_exit_offsets).
        Restore callee-save registers.
        """
        pass

    def compile_block(self, block):
        self.current_block = block.label
        if block.func.enter_block == block:
            self.compile_prologue(block)
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
        blocks = {block.label: block for block in self.ir}  # map block label to block object
        for label, (head_instrs, tail_instrs, term_instr) in self.block_instrs.items():
            if blocks[label] == blocks[label].func.enter_block:  # add label for function entry
                self.func_entry_offsets[blocks[label].func.name] = len(self.instrs)
            if blocks[label] == blocks[label].func.exit_block:
                self.func_exit_offsets[blocks[label].func.name] = len(self.instrs)
            self.block_offsets[label] = len(self.instrs)
            self.instrs.extend(head_instrs)
            self.instrs.extend(tail_instrs)
            self.instrs.extend(term_instr)
        # After this, subclasses can use self.block_offsets to link the correct jump addresses

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

    def emit_immediate(self, immediate, into, block=None):
        raise NotImplementedError()
