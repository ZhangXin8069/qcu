pushd /public/home/zhangxin/qcu
bash ./make.sh
pushd ./test
rm log_*
mpirun -n 4 python ./test.nccl.dslash.qcu-np4.py
popd
popd

