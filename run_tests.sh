#!/bin/bash
set -e

# use the python environement
source venv-activate.sh

# run all tests at first serial, then using two MPI prosesses
python -m pytest --setup-show --log-cli-level=INFO
mpirun -np 2 python -m mpi4py -m pytest --setup-show --log-cli-level=INFO