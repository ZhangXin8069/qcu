bash ./make.sh
pushd ./test
rm log_*
mpirun -n 4 python ./test.bistabcg-np4.py
popd

