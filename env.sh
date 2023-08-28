# init
_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo 'HOME:'${_HOME}
_NAME=$(basename "$0")
name='test'
work_name="test"
tmp_name="tmp"
work_path=${_HOME}/${work_name}
tmp_path=${_HOME}/${tmp_name}

# source
## mkdir
mkdir ${_HOME}/bin -p
mkdir ${_HOME}/include -p
mkdir ${_HOME}/lib -p
mkdir ${_HOME}/scripts -p
mkdir ${_HOME}/test -p
mkdir ${_HOME}/tmp -p
mkdir ${_HOME}/build -p
mkdir ${_HOME}/doc -p

# do
## export
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${_HOME}/lib/
export PYTHONPATH=$(cd ~ && pwd)/external-libraries:$PYTHONPATH

# done
