"""
Support Routines for MPI-based scripts.
"""
import sys
from numba import jit, i2
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

@jit(nopython=True)
def crc16(data: bytearray):
    """
    A short check sum that can be used to create MPI tags. Some MPI implementations don't like large tags

    The implmentation is taken from https://stackoverflow.com/questions/35205702/calculating-crc16-in-python
    """
    crc = i2(0xFFFF)
    for i in range(0, len(data)):
        crc ^= data[i] << 8
        for j in range(0, 8):
            if (crc & 0x8000) > 0:
                crc =(crc << 1) ^ 0x1021
            else:
                crc = crc << 1
    return crc & 0xFFFF
