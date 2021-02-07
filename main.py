#!/usr/bin/env python3

import sys
import argparse
import parser
import ssa
import dlx
import allocator
import dlx_emulator


def main():
    argparser = argparse.ArgumentParser(description="Compile a Smpl program.")
    argparser.add_argument("infile")
    argparser.add_argument("-o", "--output", type=argparse.FileType('wb'),
                           default=sys.stdout, help="Output file (stdout by default).")
    argparser.add_argument("--ir", default=False, action="store_true", help="Produce only IR graph.")
    argparser.add_argument("--dlx", default=False, action="store_true", help="Produce DLX machine code.")
    argparser.add_argument("--asm", default=False, action="store_true", help="Output assembly instead of machine code.")
    argparser.add_argument("--run", default=False, action="store_true", help="Run byte code in DLX emulator.")
    args = argparser.parse_args()
    with open(args.infile) as f:
        instring = f.read()
        inlexer = parser.SmplLexer(instring)
        inparser = parser.SmplParser(inlexer)
        tree = inparser.computation()
        ir = ssa.CompilationContext()
        tree.compile(ir)
        if args.ir:  # Only print SSA representation, do not compile
            args.output.write(ir.dot_repr().encode('ascii'))
            return 0
        if args.dlx:
            backend = dlx.DLXBackend(ir)
            backend.allocator = allocator.StackRegisterAllocator(ir, backend)
            backend.allocator.allocate()
            backend.compile()
            backend.link()
            if args.asm:
                args.output.write(backend.get_asm())
            elif args.run:
                dlx_emulator.DLX.load([instr.encode() for instr in backend.instrs])
                dlx_emulator.DLX.execute()
            else:
                output = args.output
                if output == sys.stdout:
                    output = output.buffer
                output.write(backend.get_machine_code())


if __name__ == "__main__":
    main()
