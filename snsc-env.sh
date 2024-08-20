# init
_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo 'HOME:'${_HOME}
# export
## zhangxin
export LD_LIBRARY_PATH=${_HOME}/lib:$LD_LIBRARY_PATH # if any
export PYTHONPATH=${_HOME}/lib:${PYTHONPATH}
## quda
export QUDA_ENABLE_P2P=0
export QUDA_ENABLE_TUNING=0