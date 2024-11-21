bash ./make.sh
pushd ./test
rm log_*
mpirun -n 2 python ./test.bistabcg-np2.py
popd

