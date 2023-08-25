#!/bin/bash

cd ..
source venv/Scripts/activate

scenario=$1 #"C:\Users\sorokin\Documents\Projects\testing\FOCETA\experiments\Leuven_PedestrianCrossing\Leuven_AVP_ori\ChangeModel.pb"
# echo "Read scenario: "$scenario
n_arga=`expr $# - 1`
n_dim=`expr $n_arga / 2` # get the number of dimensions
# read search space (we assume 2-2+n_dim and 5-5+n_dim  are min and upper bounds respectively)
bound_min=("${@:2:${n_dim}}") 
bound_max=("${@:5:${n_dim}}") 

# echo "Read bound_min: "${bound_min[@]}
# echo "Read bound_max: "${bound_max[@]}

file="interface_etb/output_opensbt.txt"
opensbt=$(python run.py -e "$scenario" -min ""${bound_min[@]} -max ""${bound_max[@]} > $file 2>&1 >/dev/null)
echo $(tail -n 1 $file | grep -oP '^INFO:root:critical_testcases:\s*\K.*' )
#rm interface_etb/output_opensbt.txt
 
################# 
# Example 
# $ bash test_interface.sh 2 1 2 3 11 12 12
# C:\Users\Lev\Documents\fortiss\projects\foceta\SBT-research/results/single/BNH\NSGA-II\temp\all_critical_testcases.csv