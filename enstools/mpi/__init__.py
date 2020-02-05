"""
Support Routines for MPI-based scripts.
"""
import sys
import petsc4py


def init_petsc():
    """
    Initialize the PETSc library using default command line arguments
    """
    petsc4py.init(sys.argv)
    comm = petsc4py.PETSc.COMM_WORLD
    return comm
