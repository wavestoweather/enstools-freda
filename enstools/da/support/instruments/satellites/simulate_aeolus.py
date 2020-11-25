"""
functions related to the simulation of aeolus observations. E.g. extraction of observations from model output files.
"""
from datetime import datetime
from enstools.da.support.feedback_file import FeedbackFile
from typing import Union, List


def add_aeolus_observation_from_model_output(ff_file: FeedbackFile, model_file: Union[str, List[str]],
                                             start_time: datetime, end_time: datetime,
                                             start_lon: float = 0, anomaly: float = 0):
    """
    Create AEOLUS-like observations and add them to a feedback file.

    Parameters
    ----------
    ff_file:
                the feedback file which is modified.

    model_file:
                file name ot list of file names to extract data from. These names are forwarded to `enstools.io.read`.

    start_time:
                the start time of the AEOLUS track

    end_time:
                the end time of the AEOLUS track

    start_lon:
                the longitude where the satellite with first overpass the equator (in degrees)

    anomaly:
                the distance from the equator along the satellite track in degrees
    """
