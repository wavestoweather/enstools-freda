#!/bin/bash -l
#SBATCH --cluster=kcs
#SBATCH --partition=kcs_batch
#SBATCH --ntasks=2
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --output=run_tests_batch.sh.%j.out
#SBATCH --job-name=run_tests_batch
#SBATCH --get-user-env
set -e

# use the python environement
source venv-activate.sh

# run all tests at first serial, then using two MPI prosesses
export NUMEXPR_MAX_THREADS=1
#python -m pytest -x --setup-show --log-cli-level=INFO
mpirun -np 2 python -m mpi4py -m pytest -x --setup-show --log-cli-level=INFO
