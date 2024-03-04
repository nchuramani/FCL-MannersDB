#!/bin/bash

#set -x

echo "You provided the arguments:" "$@"
echo "You provided $# arguments"

#/usr/local/cuda/bin/nvcc --version
nvcc --version

echo "Sourcing Virtual Environent"
#source ~/jupyter-env/bin/activate
source /home/nc528/rds/hpc-work/nc528/Code/FCLSocRob/FCL_venv/bin/activate

#python3 --version
# python3 -u main.py --batch_size "$1" --epochs 100
# python main.py -s "$1" -m MobileNet -n 10 -e 10 -c 2 -f 10 -p SADRA-Dataset -o output -b "$2" -a "$3" -t "$4"
python main.py -sfl "$1" -scl "$2" -m default -n 10 -e 10 -c 2 -f 2 -p "$3" -o "$7" -a "$5" -t "$4" -b "$6" -r "all"
echo "Deactivating Virtual Environment"
deactivate
