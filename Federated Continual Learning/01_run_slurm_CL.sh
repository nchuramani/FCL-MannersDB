#!/bin/bash

set -x

#module load cuda/9.0
#module unload cuda/8.0
#module load cuda/9.0 cudnn/7.1.3_cuda-9.0
nvcc --version

processor=cpu
aug=True
strategy=all
strategy_cl=all
base=all
name_job="FCL_MANNERSDB_withoutaug_CSV"
#name_job="FCL_MANNERSDB_withaug_FedLGR"
#data="SADRA-Dataset"
data="/home/nc528/rds/hpc-work/nc528/Datasets/MANNERSDB/SADRA-Dataset"
#output="output"
output="/home/nc528/rds/hpc-work/nc528/Models/FCLSocRob/Results/CSV/2Tasks/withoutaug"
echo $name_job
sbatch -J $name_job -o out_logs_${name_job}.out.log -e err_${name_job}.err.log 02_slurm_script_CL.script ${strategy} ${strategy_cl} ${data} ${processor} ${aug} ${base} ${output}

