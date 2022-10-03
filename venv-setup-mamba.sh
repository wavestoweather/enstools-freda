#!/bin/bash
set -e
source venv-functions.sh

# version for eccodes
EC_VERSION=2.26.0
EC_MINOR_VERSION=2

# install all dependencies using mamba
# install micromamba
if [[ ! -f micromamba ]] ; then
    ARCH=$(uname -m)
    OS=$(uname)

    if [[ "$OS" == "Linux" ]]; then
        PLATFORM="linux"
        if [[ "$ARCH" == "aarch64" ]]; then
            ARCH="aarch64";
        elif [[ $ARCH == "ppc64le" ]]; then
            ARCH="ppc64le";
        else
            ARCH="64";
        fi		
    fi

    if [[ "$OS" == "Darwin" ]]; then
        PLATFORM="osx";
        if [[ "$ARCH" == "arm64" ]]; then
            ARCH="arm64";
        else
            ARCH="64"
        fi
    fi

    mkdir -p venv
    curl -Ls https://micro.mamba.pm/api/micromamba/$PLATFORM-$ARCH/latest | tar -xvj -C $PWD/venv --strip-components=1 bin/micromamba
fi

# install dependencies in new mamba environment
if [[ ! -d venv/envs/freda ]] ; then
    activate_mamba
    micromamba create -c conda-forge -n freda
    micromamba install -c conda-forge -n freda -f requirements-mamba.txt eccodes=$EC_VERSION --yes
fi

# install dwd grib definitions
if [[ ! -d venv/eccodes_definitions ]] ; then
    # download file from DWD open data
    wget http://opendata.dwd.de/weather/lib/grib/eccodes_definitions.edzw-${EC_VERSION}-${EC_MINOR_VERSION}.tar.bz2 -O venv/eccodes_definitions.tar.bz2

    # get name of definition folder
    deffolder=$(tar -tf venv/eccodes_definitions.tar.bz2 | head -1 | xargs basename)
    echo $deffolder

    # extract
    mkdir -p venv/eccodes_definitions
    tar --directory venv/eccodes_definitions --strip-components=1 -xf venv/eccodes_definitions.tar.bz2
    rm venv/eccodes_definitions.tar.bz2
fi
