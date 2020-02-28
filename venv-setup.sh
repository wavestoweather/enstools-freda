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
    # use the python module only to create the virtual environement
    module load python
    python3 -m venv --system-site-packages --prompt nda venv
    module unload python
fi

# activate the new environement
source venv/bin/activate

# install all requirements
pip install -r requirements.txt

# install jupyter kernel
module load python
ipython kernel install --user --name enstools-nda
module unload python

# override settings to use the venv-kernel.sh script
cat > ${HOME}/.local/share/jupyter/kernels/enstools-nda/kernel.json << EOF
{
 "argv": [
  "${PWD}/venv-kernel.sh",
  "{connection_file}"
 ],
 "display_name": "enstools-nda",
 "language": "python"
}
EOF

# install the nda-package editable into the environment
pip install -e .
