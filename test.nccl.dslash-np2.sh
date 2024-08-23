pushd /public/home/zhangxin/qcu
bash ./make.sh
pushd ./test
rm log_*
mpirun -n 2 python ./test.nccl.dslash.qcu-np2.py
popd
popd

