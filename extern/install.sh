pushd ./eigen
# (arpack-ng required......)
apt install libeigen3-dev 
popd

pushd ./arpack-ng
sh bootstrap
./configure --enable-mpi --enable-eigen
make -j$(nproc)
make check
# make install # /usr/local/lib 
popd