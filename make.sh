bash ./clean.sh
source ./env.sh
cmake .
pushd ./build
make
popd
