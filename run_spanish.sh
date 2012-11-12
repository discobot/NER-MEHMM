#!/bin/bash

export MAXENTPATH=($(pwd)/maxent/python/build/lib.*)
export PYTHONPATH=${MAXENTPATH}:${PYTHONPATH}
echo "=== PYTHONPATH=${PYTHONPATH}" >&2

exec python ./run.py "$@"
