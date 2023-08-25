#!/bin/bash

# example script to trigger OpenSBT via a shell call

# input

SCENARIO=1 # the SUT/Scenario/Experiment to be executed; we can also pass directly a file to the scenario: 
           # number corresponds to default experiments defined in OpenSBT

MIN=(1 2) # the lower bounds of the search space
MAX=(11 13)  # the upper bounds of the search space

# output: path to the file with all identified critical test cases

#######################

echo $(./interface.sh $SCENARIO "${MIN[@]}" "${MAX[@]}" )

################# 
# Test 
# $ bash test_interface.sh 
# C:\Users\Lev\Documents\fortiss\projects\foceta\SBT-research/results/single/BNH\NSGA-II\temp\all_critical_testcases.csv