#!/bin/bash
set -e
# create virtual environments with all dependencies

# some dependencies are taken from the module system
module purge
module load $(cat modules.txt)

# set petsc dir to dir + architecture. This is required for petsc4py to build
if [[ -e $PETSC_DIR/$PETSC_ARCH/include/petsc.h ]] ; then
    export PETSC_DIR=$PETSC_DIR/$PETSC_ARCH
fi

# create a new environment if not yet done
if [[ ! -d venv ]] ; then
    # use the python module only to create the virtual environement
    module load python
    python3 -m venv --system-site-packages --prompt nda venv
    module unload python
fi

# activate the new environement
source venv/bin/activate

# install all requirements
pip install -r requirements.txt
