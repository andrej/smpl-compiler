#!/usr/bin/env python3

import sys
import argparse
import parser
import ssa


def main():
    argparser = argparse.ArgumentParser(description="Compile a Smpl program.")
    argparser.add_argument("infile")
    args = argparser.parse_args()
    with open(args.infile) as f:
        instring = f.read()
        inlexer = parser.SmplLexer(instring)
        inparser = parser.SmplParser(inlexer)
        tree = inparser.computation()
        ir = ssa.CompilationContext()
        tree.compile(ir)
        print(ir.dot_repr())
        for root_block in ir.root_blocks:
            sys.stderr.write("\n")
            sys.stderr.write("---\n")
            for local_name, local_op in root_block.locals_op.items():
                sys.stderr.write("{0:15s}: {1:s}\n".format(local_name, str(local_op)))
            sys.stderr.write("---\n")
            sys.stderr.write("\n")
        #res = tree.run()
        #print(res)


if __name__ == "__main__":
    main()
