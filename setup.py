# install the NDA package
from setuptools import setup
import re
import sys


def get_version():
    """
    read version string from enstools package without importing it

    Returns
    -------
    str:
            version string
    """
    with open("enstools/da/freda/__init__.py") as f:
        for line in f:
            match = re.search('__version__\s*=\s*"([a-zA-Z0-9_.]+)"', line)
            if match is not None:
                return match.group(1)


# only print the version and exit?
if len(sys.argv) == 2 and sys.argv[1] == "--get-version":
    print(get_version())
    exit()

# perform the actual install operation
setup(name="enstools-freda",
      version=get_version(),
      author="Robert Redl and Yvonne Ruckstuhl",
      author_email="robert.redl@lmu.de",
      packages=["enstools.da", "enstools.mpi"],
      namespace_packages=['enstools'],
      install_requires=[
                "enstools",
                "mpi4py",
                "petsc4py",
                "scikit-learn"],
      entry_points={
          'console_scripts': ['freda-cli=enstools.da.freda.cli:main'],
      },
      # options={
      #     'build_scripts': {'executable': '/usr/bin/python3'},
      # },
)
