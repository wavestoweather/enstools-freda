#!/bin/bash -l
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=run_tests_batch.sh.%j.out
#SBATCH --job-name=run_tests_batch
set -e

# use the python environement
if [[ -f venv/micromamba ]] ; then
    source venv-activate-mamba.sh
else
    source venv-activate.sh
fi

# run all tests at first serial, then using two MPI prosesses
export NUMEXPR_MAX_THREADS=1
#python -m pytest -x --setup-show --log-cli-level=INFO
mpirun -np 4 python -m mpi4py -m pytest -x --setup-show --log-cli-level=INFO
