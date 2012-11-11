#!/bin/bash
set -e

[[ -z "$1" ]] && echo "Please, specify canonical result" && exit 1
[[ -z "$2" ]] && echo "Please, specify computed result" && exit 2

./meld.py "$1" "$2" | ./eval.pl

