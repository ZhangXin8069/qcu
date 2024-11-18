#!/bin/bash 
#SBATCH --job-name=ssub
#SBATCH --partition=na100-ins
#SBATCH --nodes=1
#SBATCH -n 8
#SBATCH --time=00-00:30:00
#SBATCH --output=ssub.out
#SBATCH --error=ssub.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhangxin8069@qq.com
#SBATCH --gres=gpu:8
#SBATCH -w "gpu025"
source /public/home/zhangxin/env.sh
bash ./test.clover.dslash.qcu-np8.sh