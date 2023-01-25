#!/bin/bash
set -e

# use the python environement
if [[ -f venv/etc/profile.d/micromamba.sh ]] ; then
    source venv-activate-mamba.sh
else
    source venv-activate.sh
fi

# at LRZ, we need to start MPI processes be forking to run tests on the login node
if [[ $(get_site) == "lrz.de" ]] ; then
    export I_MPI_HYDRA_BOOTSTRAP=fork
    export SCRATCH=$TMPDIR
fi

# run all tests at first serial, then using two MPI prosesses
export NUMEXPR_MAX_THREADS=1
python -m pytest -x --setup-show --log-cli-level=INFO
mpirun -np 2 python -m mpi4py -m pytest -x --setup-show --log-cli-level=INFO
