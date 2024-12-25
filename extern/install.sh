pushd ./arpack-ng
sh bootstrap
./configure --enable-mpi
make
make check
sudo make install
popd