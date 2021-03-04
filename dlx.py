"""
DLX code generation from our SSA IR. Contains classes for each individual
instruction type that DLX has, as well as a DLX backend that iterates
over the IR instructions and produces corresponding DLX instructions as
it goes. Uses a provided register allocator object to translate the operands
of the IR to actual machine registers and memory locations; in turn also
provides the register allocator with methods for it to call back to generate
memory spills/loads.

Author: André Rösti
"""
import backend
import ssa


class Instruction:

    def __init__(self, opcode, mnemonic, *ops):
        self.opcode = opcode
        self.mnemonic = mnemonic
        self.ops = ops
        self.jump_label = None  # during linking, only used for BSR, JSR, RET
        self.jump_to_entry = None  # replace arg during linking to jump to func prologue
        self.jump_to_exit = None  # replace operand during linking to jump to func epilogue

    def __repr__(self):
        return self.get_assembly()

    def __bytes__(self):
        encoded = self.encode()
        return bytes((encoded>>(i*8))&255 for i in reversed(range(4)))

    def make(self, *ops):
        obj = self.__class__(self.opcode, self.mnemonic, *ops)
        return obj

    def encode(self):
        Instruction._check_overflow(self.opcode, 6)
        return self.opcode << (32-6)

    def get_assembly(self):
        return "{} {} {}".format(self.mnemonic,
                                 ", ".join(str(op) for op in self.ops),
                                 "({})".format(self.jump_label) if self.jump_label else "")

    @staticmethod
    def _check_overflow(val, max_bits):
        assert val < 1 << max_bits


class F1Instruction(Instruction):

    def encode(self):
        a, b, c = self.ops
        # Encode quantity c as 16-bit twos complement number
        if c < 0:
            c = 2**16 - abs(c)
        Instruction._check_overflow(a, 5)
        Instruction._check_overflow(b, 5)
        Instruction._check_overflow(c, 16)
        out = super().encode()
        out |= a << (32-6-5)
        out |= b << (32-6-5-5)
        out |= c
        return out


class F2Instruction(F1Instruction):

    def encode(self):
        a, b, c = self.ops
        Instruction._check_overflow(c, 5)
        return super().encode()

    def make_immediate(self):
        if not 0 <= self.opcode <= 14:
            raise Exception()
        return self.__class__(self.opcode+16, self.mnemonic+"I", self.ops)


class F3Instruction(Instruction):

    def encode(self):
        c, = self.ops
        Instruction._check_overflow(c, 26)
        out = super().encode()
        out |= c
        return out


class INSTRUCTIONS:

    ADD = F2Instruction( 0, "ADD")
    SUB = F2Instruction( 1, "SUB")
    MUL = F2Instruction( 2, "MUL")
    DIV = F2Instruction( 3, "DIV")
    MOD = F2Instruction( 4, "MOD")
    CMP = F2Instruction( 5, "CMP")
    OR  = F2Instruction( 8, "OR")
    AND = F2Instruction( 9, "AND")
    BIC = F2Instruction(10, "BIC")
    XOR = F2Instruction(11, "XOR")
    LSH = F2Instruction(12, "LSH")
    ASH = F2Instruction(13, "ASH")
    CHK = F2Instruction(14, "CHK")

    ADDI = F1Instruction(16, "ADDI")
    SUBI = F1Instruction(17, "SUBI")
    MULI = F1Instruction(18, "MULI")
    DIVI = F1Instruction(19, "DIVI")
    MODI = F1Instruction(20, "MODI")
    CMPI = F1Instruction(21, "CMPI")
    ORI  = F1Instruction(24, "ORI")
    ANDI = F1Instruction(25, "ANDI")
    BICI = F1Instruction(26, "BICI")
    XORI = F1Instruction(27, "XORI")
    LSHI = F1Instruction(28, "LSHI")
    ASHI = F1Instruction(29, "ASHI")
    CHKI = F1Instruction(30, "CHKI")

    LDW = F1Instruction(32, "LDW")
    LDX = F2Instruction(33, "LDX")
    POP = F1Instruction(34, "POP")
    STW = F1Instruction(36, "STW")
    STX = F2Instruction(37, "STX")
    PSH = F1Instruction(38, "PSH")

    BEQ = F1Instruction(40, "BEQ")
    BNE = F1Instruction(41, "BNE")
    BLT = F1Instruction(42, "BLT")
    BGE = F1Instruction(43, "BGE")
    BLE = F1Instruction(44, "BLE")
    BGT = F1Instruction(45, "BGT")

    BSR = F1Instruction(46, "BSR")
    JSR = F3Instruction(48, "JSR")
    RET = F2Instruction(49, "RET")

    RDD = F2Instruction(50, "RDD")
    WRD = F2Instruction(51, "WRD")
    WRH = F2Instruction(52, "WRH")
    WRL = F1Instruction(53, "WRL")


class DLXBackend(backend.Backend):
    """
    Compile a DLX instruction stream from the input SSA instructions.
    Uses the provided register allocator to resolve IR operands to actual
    registers and memory locations. Provides methods for the register
    allocator to emit spill and load code.
    """

    RES_REG = 5
    ZERO_REG = 0
    RET_ADD_REG = 31
    GLOBAL_MEM_PTR_REG = 30
    STACK_PTR_REG = 29
    FRAME_PTR_REG = 28

    WORD_SIZE = 4
    STACK_SIZE = 0xFFF

    CALLEE_SAVE = [1, 2]

    def __init__(self, ir):
        super().__init__(ir)
        self.heap_height = 0

    def compile_prologue(self, func_block):
        if func_block.func.is_main:
            return  # no prelude required for main func
        #  --- high addr ---
        #          *  LOCAL 1
        #          *  LOCAL 2
        #          *  ...
        # caller   *  PARAM 1
        #          *  PARAM 2
        #          *  ...
        #          *  OLD RET ADDR
        #          / caller's old frame ptr <--- frame ptr
        #          / CALLEE SAVED REG 1
        #          / CALLEE SAVED REG 2
        #          / ...
        #          / LOCAL 1
        # callee   / LOCAL 2
        #          / ...
        #          / LOCAL N           <--- stack ptr
        #  --- low  addr ---

        # Update frame pointer to point to bottom of calling functions stack
        # and store old frame pointer
        self.emit(INSTRUCTIONS.PSH.make(self.RET_ADD_REG, self.STACK_PTR_REG, -self.WORD_SIZE))
        self.emit(INSTRUCTIONS.PSH.make(self.FRAME_PTR_REG, self.STACK_PTR_REG, -self.WORD_SIZE))
        self.emit_move(self.STACK_PTR_REG, self.FRAME_PTR_REG)

        # Save registers
        for i, reg in enumerate(self.CALLEE_SAVE):
            self.emit(INSTRUCTIONS.STW.make(reg, self.FRAME_PTR_REG, -(i+1)*self.WORD_SIZE))

    def compile_epilogue(self, func_block):
        # Restore registers
        for i, reg in enumerate(self.CALLEE_SAVE):
            self.emit(INSTRUCTIONS.LDW.make(reg, self.FRAME_PTR_REG, -(i+1)*self.WORD_SIZE))
        # restore old frame and stack pointers
        self.emit_move(self.FRAME_PTR_REG, self.STACK_PTR_REG)
        self.emit(INSTRUCTIONS.POP.make(self.FRAME_PTR_REG, self.STACK_PTR_REG, +self.WORD_SIZE))
        self.emit(INSTRUCTIONS.POP.make(self.RET_ADD_REG, self.STACK_PTR_REG, +self.WORD_SIZE))
        self.emit(INSTRUCTIONS.RET.make(0, 0, self.RET_ADD_REG))

    def compile_init(self):
        # We iterate over all memory allocation calls to determine the heap height;
        # we then allocate the stack to start right below that, to make maximum use
        # of memory
        heap_size = 0
        for block in self.ir:
            for instr in block.instrs:
                if instr.instr != "alloca":
                    continue
                assert isinstance(instr.ops[0], ssa.ImmediateOp)  # Dynamic memory allocation currently not supported
                sz = instr.ops[0].val
                heap_size += sz
        # subtract heap size from global memory pointer address
        self.emit(INSTRUCTIONS.ADDI.make(self.FRAME_PTR_REG, self.GLOBAL_MEM_PTR_REG, -heap_size*4))
        self.emit_move(self.FRAME_PTR_REG, self.STACK_PTR_REG)

        # jump to main function
        #dlx_instr = INSTRUCTIONS.JSR.make(0)
        #dlx_instr.jump_to_entry = "main"
        #self.emit(dlx_instr)
        #self.emit(INSTRUCTIONS.RET.make(0, 0, 0))

    def compile_operand(self, op: ssa.Op, context: ssa.BasicBlock, into=1, block=None, back=False):
        emit_fun = self.emit if not back else self.emit_back
        if isinstance(op, ssa.ImmediateOp):
            if op.val < 1 << 16:
                emit_fun(INSTRUCTIONS.ADDI.make(into, self.ZERO_REG, op.val),
                         block=block)
                return into
            else:
                raise Exception("Immediate values may be at most < 2**16")
        elif isinstance(op, ssa.InstructionOp):
            op_reg = self.allocator.access(op.i, into=into, block=block, back=back)
            return op_reg
        elif isinstance(op, ssa.ArgumentOp):
            # ABI: We pass arguments on the stack.
            # They are in reverse order just below the return address. The return
            # address is the last thing stored by the calling function on its stack,
            # and the frame pointer points to the last item of the caller stack.
            # Hence, arguments start at FRAME_PTR+1.
            idx = context.func.arg_names.index(op.name)
            offset = 1 + (len(context.func.arg_names) - idx)
            emit_fun(INSTRUCTIONS.LDW.make(into, self.FRAME_PTR_REG, offset*self.WORD_SIZE),
                     block=block)
            return into
        return None  # Label arguments will be replaced in the linking phase

    def compile_block(self, block):
        super().compile_block(block)
        # Make sure fall-through leads to correct block
        # TODO remove for blocks if succ is directly fall-through
        if block.succs:
            dlx_instr = INSTRUCTIONS.BSR.make(0, 0, 0)
            dlx_instr.jump_label = block.succs[0].label
            self.emit_term(dlx_instr)

    def compile_instr(self, instr: ssa.Instruction, context: ssa.BasicBlock):
        # translate the neg(x) op into sub(0, x)
        if instr.instr == "neg":
            instr.instr = "sub"
            instr.ops = [ssa.ImmediateOp(0)] + instr.ops

        # Simple F1 instructions
        arith_f1_instrs = {
            "add": INSTRUCTIONS.ADD,
            "adda": INSTRUCTIONS.ADD,  # TODO actually implement LDX, for now its ADD followed by LDW
            "sub": INSTRUCTIONS.SUB,
            "mul": INSTRUCTIONS.MUL,
            "div": INSTRUCTIONS.DIV,
            "cmp": INSTRUCTIONS.CMP
        }

        # For most instructions, we first have to compile the operands.
        # We do not do it here for arithmetic F1 instructions, since we might want to use
        # immediate variants of these instructions instead if one of the operands are immediate.
        # We also do not do it for phi nodes, since those are a simple move of a previous instr,
        # or alloca, where we handle it ourselves.
        # We do not do it for call instructions, because there we can have an unbounded number
        # of arguments, that we do not want to all put into registers. Instead, we push them
        # on the stack in this instruction-specific code.
        if instr.instr not in {"phi", "alloca", "call", "return"} and instr.instr not in arith_f1_instrs:
            ops = [self.compile_operand(op, context=context, into=i+1) for i, op in enumerate(instr.ops)]
            # ops will contain register numbers / immediate values for all operands

        # Conditional jump F1 instructions
        jump_f1_instrs = {
            "beq": INSTRUCTIONS.BEQ,
            "ble": INSTRUCTIONS.BLE,
            "blt": INSTRUCTIONS.BLT,
            "bge": INSTRUCTIONS.BGE,
            "bgt": INSTRUCTIONS.BGT,
            "bne": INSTRUCTIONS.BNE
        }

        # Simple F1 instructions
        if instr.instr in arith_f1_instrs:
            dlx_instr = arith_f1_instrs[instr.instr]
            if (isinstance(instr.ops[1], ssa.ImmediateOp)
                    or isinstance(instr.ops[0], ssa.ImmediateOp) and instr.instr in {"add", "sub"}):
                dlx_instr = dlx_instr.make_immediate()
                ssa_op_a, ssa_op_b = instr.ops
                if isinstance(ssa_op_a, ssa.ImmediateOp) and instr.instr in {"add", "mul"}:
                    ssa_op_a, ssa_op_b = ssa_op_b, ssa_op_a
                elif isinstance(ssa_op_a, ssa.ImmediateOp) and instr.instr == "sub":
                    ssa_op_a, ssa_op_b = ssa_op_a, -ssa_op_b
                ops = [self.compile_operand(ssa_op_a, context=context, into=1),
                       ssa_op_b.val]
            else:
                # Neither is immediate, just regularly compile
                ops = [self.compile_operand(op, context=context, into=i+1) for i, op in enumerate(instr.ops)]
            self.emit(dlx_instr.make(self.RES_REG, *ops))
            self.allocator.store(instr.i, self.RES_REG)

        elif instr.instr in jump_f1_instrs:
            dlx_instr = jump_f1_instrs[instr.instr].make(ops[0], 0, 0)
            assert isinstance(instr.ops[1], ssa.LabelOp)
            dlx_instr.jump_label = instr.ops[1].label
            self.emit_term(dlx_instr)

        elif instr.instr == "bra":
            dlx_instr = INSTRUCTIONS.BSR.make(0, 0, 0)
            assert isinstance(instr.ops[0], ssa.LabelOp)
            dlx_instr.jump_label = instr.ops[0].label
            self.emit_term(dlx_instr)

        elif instr.instr == "store":
            self.emit_heap_store(ops[1], ops[0])

        elif instr.instr == "load":
            self.emit_heap_load(ops[0], self.RES_REG)
            self.allocator.store(instr.i, self.RES_REG)

        elif instr.instr == "alloca":
            assert isinstance(instr.ops[0], ssa.ImmediateOp)  # Dynamic memory allocation currently not supported
            sz = instr.ops[0].val
            self.heap_height -= sz
            self.emit(INSTRUCTIONS.ADDI.make(self.RES_REG, self.ZERO_REG, self.heap_height),
                      block=context.label)
            self.allocator.store(instr.i, self.RES_REG)

        elif instr.instr == "call":
            assert isinstance(instr.ops[0], ssa.FunctionOp)
            # built-in functions
            if instr.ops[0].func == "InputNum":
                self.emit(INSTRUCTIONS.RDD.make(self.RES_REG, 0, 0))
                self.allocator.store(instr.i, self.RES_REG)
                return
            elif instr.ops[0].func == "OutputNum":
                self.compile_operand(instr.ops[1], context=context, into=self.RES_REG)
                self.emit(INSTRUCTIONS.WRD.make(0, self.RES_REG, 0))
                return
            elif instr.ops[0].func == "OutputNewLine":
                self.emit(INSTRUCTIONS.WRL.make(0, 0, 0))
                return

            # UPDATE STACK POINTER
            # since we do not actually use our stack pointer (instead use absolute
            # offsets above frame pointer) we need to update it here so the callee
            # knows where to start writing its values on the stack.
            n_callee_save = len(self.CALLEE_SAVE)
            stack_height = self.allocator.func_stack_heights[context.func.name] + n_callee_save
            self.emit(INSTRUCTIONS.ADDI.make(self.STACK_PTR_REG, self.FRAME_PTR_REG, -stack_height*self.WORD_SIZE))  # R29 = R28 - this func stack height

            # PUSH ARGUMENTS ONTO STACK
            for arg in instr.ops[1:]:
                arg_reg = self.compile_operand(arg, context=context, into=self.RES_REG)
                self.emit(INSTRUCTIONS.PSH.make(arg_reg, self.STACK_PTR_REG, -self.WORD_SIZE))

            # JUMP
            # actual function call: jump to label, storing return address in R31
            dlx_instr = INSTRUCTIONS.JSR.make(0)
            dlx_instr.jump_to_entry = instr.ops[0].func
            self.emit(dlx_instr)

            # STORE RETURN VALUE
            # Values are passed back in register RES_REG
            self.allocator.store(instr.i, self.RES_REG)

        elif instr.instr == "return":
            if instr.ops:  # return values are passed in register RES_REG
                self.compile_operand(instr.ops[0], context=context, into=self.RES_REG)
            self.compile_epilogue(context)
            #dlx_instr = INSTRUCTIONS.JSR.make(0)
            #dlx_instr.jump_to_exit = context.func.name

        elif instr.instr == "phi":
            pred_a, op_a, pred_b, op_b = instr.ops
            if op_a != op_b:
                for (pred, op) in [(pred_a, op_a), (pred_b, op_b)]:
                    op_compiled = self.compile_operand(op, context=context,
                                                       into=7,
                                                       block=pred.label, back=True)
                    # TODO check whether op compiled != access of op
                    self.allocator.store(instr.i, op_compiled,
                                         block=pred.label, back=True)

        elif instr.instr == "end":
            self.emit_term(INSTRUCTIONS.RET.make(0, 0, 0))

    def link(self):
        super().link()
        for i, instr in enumerate(self.instrs):
            if not instr.jump_label and not instr.jump_to_entry and not instr.jump_to_exit:
                continue
            target = None
            target_label = instr.jump_label
            lookup_map = self.block_offsets
            if instr.jump_to_entry:
                target_label = instr.jump_to_entry
                lookup_map = self.func_entry_offsets
            elif instr.jump_to_exit:
                target_label = instr.jump_to_exit
                lookup_map = self.func_exit_offsets
            if target_label not in lookup_map:
                raise Exception("Unknown symbol {}".format(instr.jump_label))
            # Jump instructions have their target as arg 3 (c)
            ops = list(instr.ops)
            if instr.opcode == INSTRUCTIONS.JSR.opcode:
                target = lookup_map[target_label] * self.WORD_SIZE  # (absolute offset)
            else:
                target = lookup_map[target_label] - i
            ops[-1] = target
            instr.ops = tuple(ops)
            self.instrs[i] = instr

    def emit_stack_load(self, offset, into, block=None, back=False):
        """
        We actually use the frame pointer as the base address for our stack
        loads and writes. This allows us to use "absolute" offsets within
        the function, whereas the stack pointer may move to stay on top of
        the stack.
        """
        emit_fun = self.emit if not back else self.emit_back
        n_callee_save = len(self.CALLEE_SAVE)
        offset += n_callee_save + 1  # adjust for the portion of the stack used for calle save registers in prologue
        # Memory is byte addressed and one word is four bytes
        emit_fun(INSTRUCTIONS.LDW.make(into, self.FRAME_PTR_REG, -offset*self.WORD_SIZE),
                 block=block)

    def emit_stack_store(self, addr_offs, val_reg, block=None, back=False):
        emit_fun = self.emit if not back else self.emit_back
        n_callee_save = len(self.CALLEE_SAVE)
        addr_offs += n_callee_save + 1
        emit_fun(INSTRUCTIONS.STW.make(val_reg, self.FRAME_PTR_REG, -addr_offs*self.WORD_SIZE),
                 block=block)

    def emit_heap_load(self, addr_offs_reg, into, block=None, back=False):
        emit_fun = self.emit if not back else self.emit_back
        emit_fun(INSTRUCTIONS.LDX.make(into, self.GLOBAL_MEM_PTR_REG, addr_offs_reg),
                 block=block)

    def emit_heap_store(self, addr_offs_reg, val_reg):
        self.emit(INSTRUCTIONS.STX.make(val_reg, self.GLOBAL_MEM_PTR_REG, addr_offs_reg))

    def emit_move(self, from_reg, to_reg):
        self.emit(INSTRUCTIONS.ADDI.make(to_reg, from_reg, 0))

    def emit_immediate(self, immediate, into, block=None):
        self.emit(INSTRUCTIONS.ADDI.make(into, self.ZERO_REG, immediate),
                  block=block)