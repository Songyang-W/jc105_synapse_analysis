#!/bin/bash -l
#this makes it so you'll get an email at the (b)eginning of the job, (e)nd of the job, and on an (a)bort of the job
#$ -m ea
#this merges output and error files into one file
#$ -j y
#this sets the project for the script to be run under
#$ -P jchenlab
#$ -e log_files/
#Specify the time limit
#$ -l h_rt=1:00:00
#$ -pe omp 4
#$ -l mem_per_core=2G

module load python3/3.10.12
source /projectnb/jchenlab/venvs/cave_analysis/bin/activate

cd /net/claustrum/mnt/data/Dropbox/Chen\ Lab\ Dropbox/Chen\ Lab\ Team\ Folder/Projects/Connectomics/Analysis/cave_client_analysis/jc105_synapse_analysis/
add_mat_version_lookuptable.py
