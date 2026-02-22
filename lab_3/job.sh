#!/bin/bash

#SBATCH -J parprog
#SBATCH -e /work/scratch/kurse/kurs00093/<TUID>/stderr/stderr.parprog.%j.txt
#SBATCH -o /work/scratch/kurse/kurs00093/<TUID>/stdout/stdout.parprog.%j.txt
#SBATCH -C avx512
#SBATCH -n 1
#SBATCH --mem-per-cpu=1024
#SBATCH --time=5
#SBATCH --cpus-per-task=96
#SBATCH -c 96
#SBATCH -A kurs00093
#SBATCH -p kurs00093
#SBATCH --gres=gpu:v100
#SBATCH --reservation=kurs00093

module purge
module load git/2.47 cmake/3.26 gcc/11.2 cuda/11.8

echo "This is job $SLURM_JOB_ID"
cd build
bin/lab_test
