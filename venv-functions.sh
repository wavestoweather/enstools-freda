# source this file!
if [[ "${BASH_SOURCE[0]}" == "${0}" ]] ; then
    echo "ERROR: this file is only sourced by other venv-* files!"
    exit 1
fi

# get the local side
function get_site() {
    RESULT=$(hostname -d | sed 's/\./ /g' | rev | awk '{ print $1"."$2}' | rev)
    echo $RESULT
}

# some dependencies are taken from the module system
function load_modules() {
    module purge
    case $(get_site) in
        lrz.de)
            LRZ_MODULE_ROOT=/dss/dssfs02/lwp-dss-0001/uh211/uh211-dss-0000/ru24sul/
            module purge
            module use $LRZ_MODULE_ROOT/modulefiles
            module load $(cat modules-lrz.txt)
            export CFLAGS=-Wl,-rpath-link=/lrz/sys/intel/impi2019u6/impi/2019.6.154/intel64/libfabric/lib
            ;;
        *)
            module load $(cat modules.txt)
            ;;
    esac
    module list
}
