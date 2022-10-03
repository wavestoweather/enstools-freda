#!/bin/bash
set -e
# create virtual environments with all dependencies

# some dependencies are taken from the module system
source venv-functions.sh
load_modules

# set petsc dir to dir + architecture. This is required for petsc4py to build
if [[ -e $PETSC_DIR/$PETSC_ARCH/include/petsc.h ]] ; then
    export PETSC_DIR=$PETSC_DIR/$PETSC_ARCH
fi

# create a new environment if not yet done
if [[ ! -d venv ]] ; then
    python3 -m venv --prompt freda venv
fi

# activate the new environement
source venv/bin/activate

# are we using intel compilers?
if which icc &> /dev/null ; then
    echo "INFO: using intel compilers!"
    export CC=icc
    export CXX=icpc
    export FC=ifort
    export MPICC=mpiicc
    export MPICXX=mpiicpc
    export MPIF90=mpiifort
fi

# install all requirements
pip install --upgrade pip wheel
pip install 'numpy<1.21.0'
pip install --no-binary :all: mpi4py
export CFLAGS="-DACCEPT_USE_OF_DEPRECATED_PROJ_API_H"
pip install -r requirements.txt

# install jupyter kernel
module load python
ipython kernel install --user --name enstools-freda
module unload python

# override settings to use the venv-kernel.sh script
cat > ${HOME}/.local/share/jupyter/kernels/enstools-freda/kernel.json << EOF
{
 "argv": [
  "${PWD}/venv-kernel.sh",
  "{connection_file}"
 ],
 "display_name": "enstools-freda",
 "language": "python"
}
EOF

# install the freda-package editable into the environment
pip install -e .
