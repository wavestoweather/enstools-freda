# FREDA: A framework for research on ensemble data assimilation 

The Python package `enstools-freda` provides a very easy-to-use framework for 
the development of new data assimilation algorithms. New algorithms are 
implemented as plugins, often in a few lines of code, and can be easily 
exchanged. In the best case, this takes the form of a single Python function, 
which only describes the actual mathematical implementation of the algorithm. 
The framework covers the rest, including parallelization, data handling, and 
I/O. Ensemble square root and Kalman filters (EnSRF, EnKF) have been 
implemented as a first step. 

FREDA is developed within [Waves to Weather - Transregional Collaborative Research 
Project (SFB/TRR165)](https://wavestoweather.de). 

# Installation

`mamba` is the easiest way to install `enstools-freda` along with all dependencies.

    ./venv-setup-mamba.sh


# Examples

The directory `examples` contains a Jupyter Notebook that illustrates the functionality.

# Acknowledgment and license

FREDA (`enstools-freda`) is a collaborative development of the subprojects
[B6](https://www.wavestoweather.de/research_areas/phase2/b6) and 
[Z2](https://www.wavestoweather.de/research_areas/phase2/z2) within
Waves to Weather (SFB/TRR165). The development has been funded by the 
German Research Foundation (DFG).

The code is released under an [Apache-2.0 licence](./LICENSE).
