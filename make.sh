# source ./env.sh
cmake .
make -j16
mkdir ./lib -p
mv libqcu.so ./lib
bash ./clean.sh
