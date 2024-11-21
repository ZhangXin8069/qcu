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
## mpi
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
## nccl
export NCCL_DEBUG=
# export NCCL_DEBUG=INFO
# cat /proc/net/dev | awk '{i++; if(i>2){print $1}}' | sed 's/^[\t]*//g' | sed 's/[:]*$//g'
export NCCL_SOCKET_IFNAME=eth0
