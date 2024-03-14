bash ./clean.sh
pushd ../
source ./env.sh
popd
cmake .
make -j48

