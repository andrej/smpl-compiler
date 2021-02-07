import typing
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
        self.stack_height = 0
        self.offsets = {}

    def allocate(self):
        stack_height = 0
        self.offsets = {}
        for block in self.ir:
            if block.func.enter_block == block:
                # Every function has their own stack
                stack_height = 0
            for instr in block.instrs:
                if not instr.produces_output:
                    continue
                self.offsets[instr.i] = stack_height
                stack_height += 1

    def access(self, var_name, into=0, block=None, back=False):
        if var_name not in self.offsets:
            raise Exception("Access to undeclared variable {}.".format(var_name))
        self.backend.emit_stack_load(self.offsets[var_name], into,
                                     block=block, back=back)
        return into

    def store(self, var_name, value_reg, block=None, back=False):
        self.backend.emit_stack_store(self.offsets[var_name], value_reg, block=block, back=back)
