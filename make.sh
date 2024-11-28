# source ./env.sh
cmake .
make -j36
mv libqcu.so ./lib
bash ./clean.sh
