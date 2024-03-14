# init
_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo 'HOME:'${_HOME}
# export
## zhangxin
export LD_LIBRARY_PATH=${_HOME}/lib:$LD_LIBRARY_PATH # if any
export TERM=xterm-256color
export PATH=$PATH:${_HOME}/bin
export PYTHONPATH="/home/aistudio/external-libraries" # bml
## openmpi
## cuda