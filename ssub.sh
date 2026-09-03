#!/bin/bash 
#SBATCH --job-name=ssub
#SBATCH --partition=na100-ins
#SBATCH --nodes=1
#SBATCH -n 8
#SBATCH --output=ssub.out
#SBATCH --error=ssub.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhangxin8069@qq.com
#SBATCH --reservation=zhangxin_84
#SBATCH --gres=gpu:8
#SBATCH -w "gpu025"

# source /public/home/zhangxin/env.sh
# bash ./make.sh
# pushd ./test
# rm log_*
# mpirun -n 8 python ./test.nccl.bistabcg.qcu-np8.py
# popd
sleep 70000