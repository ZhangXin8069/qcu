# init
echo "There is init!"
# export
source ./env.sh
# make
cmake .
make -j$(nproc)
# clean
# make clean
rm -rf CMakeFiles
rm cmake_install.cmake
rm CMakeCache.txt
rm Makefile
rm -rf build