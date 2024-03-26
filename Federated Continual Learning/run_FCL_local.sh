#!/bin/bash


processor=cpu
aug=True
strategy=all
strategy_cl=all
base=all
name_job="FCL_MANNERSDB_withoutaug_CSV"
output="/path/to/output"
rounds=10
epochs=10
icl=2
fcl=2
reg_coef='all'
data="/path/to/dataset"


python main.py --strategy_fl ${strategy} --strategy_cl ${strategy_cl} --rounds ${rounds} --epochs ${epochs} --icl ${icl} --fcl ${fcl} --path ${data} --output ${output} --aug ${aug} --processor_type ${processor} --base ${base} --reg_coef ${reg_coef}
