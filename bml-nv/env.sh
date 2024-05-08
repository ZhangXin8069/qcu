# init
_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo 'HOME:'${_HOME}
# export
export LD_LIBRARY_PATH=
## zhangxin
export LD_LIBRARY_PATH=${_HOME}/lib:$LD_LIBRARY_PATH # if any
export TERM=xterm-256color
export PATH=${_HOME}/bin:$PATH
export PYTHONPATH="/home/aistudio/external-libraries" # bml
## openmpi
MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi
export PATH=${MPI_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export MPI_INCLUDE_PATH=${MPI_HOME}/include:$MPI_INCLUDE_PATH
export MANPATH=${MPI_HOME}/share/man:$MANPATH
### Manually set MPI include and library paths
export MPI_INCLUDE_PATH="${MPI_HOME}/include"
export MPI_CXX_LIBRARIES="${MPI_HOME}/lib/libmpi.so"
## cuda
CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib:$LD_LIBRARY_PATH
export CUDA_INCLUDE_PATH=${CUDA_HOME}/include:$CUDA_INCLUDE_PATH
export MANPATH=${CUDA_HOME}/share/man:$MANPATH