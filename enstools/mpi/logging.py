"""
Support Routines for logging in MPI-based scripts.
"""
import logging
import math
from datetime import datetime


def log_on_rank(message: str, level: int, comm=None, rank: int = 0, barrier: bool = False):
    """

    Parameters
    ----------
    message: str
            message to show

    level: int
            logging-level. Use levels from logging module

    comm: MPI communicator object.
            required to check the rank. if not given, the message is shown everywhere.

    rank: int
            number of the current rank. Use -1 for all ranks.
    """
    # wait for all processes to reach this point (gives cleaner logs)
    if comm is not None and barrier is True:
        comm.barrier()

    # use the number of other running timings for indention of the log
    pre_prefix = "--" * len(timers)
    if len(pre_prefix) > 0:
        pre_prefix += "> "

    # check the current rank of this processor
    if comm is not None:
        current_rank = comm.Get_rank()
    else:
        current_rank = None

    # are we on the correct rank?
    if current_rank is not None and rank is not None:
        if current_rank != rank and rank != -1:
            return
        else:
            width = math.floor(math.log(comm.getSize(), 10)) + 1
            prefix = f"rank={current_rank:{width}d} => "
    else:
        prefix = ""

    # show a message with the correct log-level
    logging.log(level, f"{prefix}{pre_prefix}{message}")


# dictionary for timed procedures
timers = {}


def log_and_time(message: str, level:int, isstart: bool, comm=None, rank: int = 0, barrier: bool = False):
    """
    write a log message and start timing for the given string.

    Parameters
    ----------
    message: str
            name of the function or code-block to time

    level: int
            log level to use.

    comm: MPI communicator object
            used to find the current rank

    rank: int
            rank to do logging for

    barrier: bool
            wait for all processes to reach this point.

    isstart: bool
            this is the start of the timing.
    """
    # get the current time:
    now = datetime.now()
    if isstart:
        prefix = "start timing for "
        suffix = ""
    else:
        diff = now - timers[message]
        del timers[message]
        prefix = "runtime for "
        suffix = f": {diff}"
    log_on_rank(f"{prefix}{message}{suffix}", level, comm, rank, barrier)
    if isstart:
        timers[message] = now


