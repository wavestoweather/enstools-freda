DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# source this file!
if [[ "${BASH_SOURCE[0]}" == "${0}" ]] ; then
    echo "ERROR: use: source ${BASH_SOURCE[0]}"
    exit 1
fi

# setup environment for micromamba
source ${DIR}/venv-functions.sh
activate_mamba

# activate freda environment
micromamba activate freda

# grib definitions from DWD
export ECCODES_DEFINITION_PATH=${DIR}/venv/eccodes_definitions
