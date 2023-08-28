bash ./make_clean.sh
source ./env.sh
cmake .
pushd ./build
make
popd
