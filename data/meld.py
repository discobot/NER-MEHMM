#!/usr/bin/python

import sys
import itertools

if len(sys.argv) < 3:
    print >>sys.stderr, "Please, specify arguments."
    sys.exit(1)

f1 = open(sys.argv[1])
f2 = open(sys.argv[2])

for lhs, rhs in itertools.izip(f1, f2):
    lhs = lhs.rstrip("\r\n")
    rhs = rhs.rstrip("\r\n")

    if (lhs == "" and rhs != "") or (lhs != "" and rhs == ""):
        print >>sys.stderr, "Misaligned sequences."
        sys.exit(2)

    print lhs, rhs
