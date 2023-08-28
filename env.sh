# init
_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo 'HOME:'${_HOME}
_NAME=$(basename "$0")
SM_ARCH="sm_80"
MAXRREGCOUNT="512"

# source
## mkdir
mkdir ${_HOME}/bin -p
mkdir ${_HOME}/include -p
mkdir ${_HOME}/lib -p
mkdir ${_HOME}/test -p
mkdir ${_HOME}/doc -p
mkdir ${_HOME}/refer -p

## export
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${_HOME}/lib/
export PYTHONPATH=$(cd ~ && pwd)/external-libraries:$PYTHONPATH
