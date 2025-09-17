#!/bin/bash -l
#$ -m ea
#$ -j y
#$ -P jchenlab
#$ -e log_files/
#$ -l h_rt=6:00:00
#$ -pe omp 4
#$ -l mem_per_core=2G
#$ -t 1-8   # <-- run 8 tasks in parallel

module load python3/3.10.12
source /projectnb/jchenlab/venvs/cave_analysis/bin/activate

python split_root_ids.py $SGE_TASK_ID
