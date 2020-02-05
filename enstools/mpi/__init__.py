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

def onRank0(comm):
    """
    return True if the comm object if None or if we are currently on rank 0
    """
    if comm is None or comm.Get_rank() == 0:
        return True
    else:
        return False

def isGt1(comm):
    """
    return True, if the communicator is not None and the size is larger 1
    """
    if comm is None or comm.Get_size() == 1:
        return False
    else:
        return True
