#!/usr/bin/env python3

import sys
import argparse
import parser


def main():
    argparser = argparse.ArgumentParser(description="Compile a Smpl program.")
    argparser.add_argument("infile")
    args = argparser.parse_args()
    with open(args.infile) as f:
        instring = f.read()
        inlexer = parser.SmplLexer(instring)
        inparser = parser.SmplParser(inlexer)
        tree = inparser.computation()
        #print(tree.dot_repr())
        res = tree.run()
        print(res)


if __name__ == "__main__":
    main()
