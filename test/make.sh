# source ./env.sh
cmake .
make -j$(nproc)
# mv libqcu.so ./lib
bash ./clean.sh