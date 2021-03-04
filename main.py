#!/usr/bin/env python3
"""
Smpl compiler command-line interface.

Author: André Rösti
"""

import sys
import argparse
import parser
import ssa
import dlx
import allocator
import dlx_emulator


def main():

    # Argument parsing
    argparser = argparse.ArgumentParser(description="Compile a Smpl program to abstract syntax tree, "
                                                    "SSA intermediate representation graph, DLX assembly or "
                                                    "DLX machine code.")
    argparser.add_argument("infile")
    argparser.add_argument("-o", "--output", type=argparse.FileType('wb'),
                           default=sys.stdout, help="Output file (stdout by default).")
    argparser.add_argument("--ast", default=False, action="store_true", help="Produce only AST graph.")
    argparser.add_argument("--interpret", default=False, action="store_true", help="Run interpreter on the input program instead of compiling.")
    argparser.add_argument("--ir", default=False, action="store_true", help="Produce only IR graph.")
    argparser.add_argument("--no-ce", default=False, action="store_true", help="Disable constant elimination.")
    argparser.add_argument("--no-dce", default=False, action="store_true", help="Disable dead code elimination.")
    argparser.add_argument("--no-cse", default=False, action="store_true", help="Disable common subexpression elimination.")
    argparser.add_argument("--dlx", default=False, action="store_true", help="Produce DLX machine code.")
    argparser.add_argument("--asm", default=False, action="store_true", help="Output assembly instead of machine code.")
    argparser.add_argument("--run", default=False, action="store_true", help="Run byte code in DLX emulator.")
    args = argparser.parse_args()
    if (args.asm or args.run) and not args.dlx:
        print("--asm and --run options can only be used in conjunction with --dlx option to create DLX assembly or run DLX machine code.")
        return 1
    output = args.output
    if output == sys.stdout:
        output = output.buffer

    # Read input
    with open(args.infile) as f:
        instring = f.read()
    if not instring:
        print("Input file is empty.")
        return 1

    # Parse the input
    inlexer = parser.SmplLexer(instring)
    inparser = parser.SmplParser(inlexer)
    tree = inparser.parse()
    if args.ast:
        output.write(tree.dot_repr().encode('ascii'))
        return 0
    if args.interpret:
        tree.run()
        return

    # Compile AST to intermediate SSA representation
    ir = ssa.CompilationContext(do_cse=not args.no_cse)
    tree.compile(ir)
    if not args.no_dce:
        ir.eliminate_dead_code()
    ir.print_warnings()
    if not args.no_ce:
        ir.eliminate_constants()
    if args.ir:  # Only print SSA representation, do not compile
        output.write(ir.dot_repr().encode('ascii'))
        return 0

    # Compile IR to DLX machine code or assembly
    if args.dlx:
        backend = dlx.DLXBackend(ir)
        backend.allocator = allocator.StackRegisterAllocator(ir, backend)
        backend.allocator.allocate()
        backend.compile()
        backend.link()
        if args.asm:
            output.write(backend.get_asm().encode("ascii"))
        if args.run:
            dlx_emulator.DLX.load([instr.encode() for instr in backend.instrs])
            dlx_emulator.DLX.execute()
        if not args.asm and not args.run:
            output.write(backend.get_machine_code())


if __name__ == "__main__":
    sys.exit(main())
