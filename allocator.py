"""
An abstract interface definition for register allocators, along with some implementations.

A register allocator must be provided to a Backend upon code generation. The backend will
call back to the register allocator with requests to map ssa.Ops to actual machine
registers and memory locations (for spills). To be backend-independent, the register
allocator must then call back to the appropriate methods in the backend to generate
load and store instructions.

Author: André Rösti
"""

import ssa
import backend


class Op:
    pass


class ImmediateOp(Op):

    def __init__(self, val):
        self.val = val


class RegisterOp(Op):

    def __init__(self, i):
        self.i = i


class StackOp(Op):

    def __init__(self, offset):
        self.offset = offset


class HeapOp(Op):

    def __init__(self, addr):
        self.addr = addr


class RegisterAllocator:

    def __init__(self, ir: ssa.CompilationContext, be: backend.Backend):
        self.ir: ssa.CompilationContext = ir
        self.backend: backend.Backend = be

    def allocate(self):
        """
        Scan the IR and allocate registers.
        """
        raise NotImplementedError()

    def store(self, var_name, val):
        """
        Emit code to store val into the register allocated for var_name.
        """
        raise NotImplementedError()

    def access(self, var_name):
        """
        Emit code (if necessary) to load var_name into a register; then
        return the register in which the variable was loaded.
        :param var_name:
        :return:
        """
        raise NotImplementedError()


class StackRegisterAllocator(RegisterAllocator):
    """
    The most naive register allocator. Stores and loads everything to/from memory stack.
    """

    def __init__(self, ir: ssa.CompilationContext, be: backend.Backend):
        super().__init__(ir, be)
        self.stack_offsets = {}
        self.func_stack_heights = {}
        self.heap_offsets = {}
        self.total_stack_height = 0

    def allocate(self):
        stack_height = 0
        self.stack_offsets = {}
        for block in self.ir:
            if block.func.enter_block == block:
                # Every function has their own stack
                stack_height = 0
                self.func_stack_heights[block.func.name] = 0
            for instr in block.instrs:
                if not instr.produces_output:
                    continue
                self.stack_offsets[instr.i] = stack_height
                stack_height += 1
                self.total_stack_height += 1
                self.func_stack_heights[block.func.name] += 1

    def access(self, var_name, into=0, block=None, back=False):
        if var_name not in self.stack_offsets:
            raise Exception("Access to undeclared variable {}.".format(var_name))
        self.backend.emit_stack_load(self.stack_offsets[var_name], into,
                                     block=block, back=back)
        return into

    def store(self, var_name, value_reg, block=None, back=False):
        self.backend.emit_stack_store(self.stack_offsets[var_name], value_reg, block=block, back=back)
