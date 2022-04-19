# source this file!
if [[ "${BASH_SOURCE[0]}" == "${0}" ]] ; then
    echo "ERROR: use: source ${BASH_SOURCE[0]}"
    exit 1
fi

file_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# load required modules
source ${file_dir}/venv-functions.sh
load_modules

# activate python environment
source ${file_dir}/venv/bin/activate
export HDF5_DISABLE_VERSION_CHECK=2
