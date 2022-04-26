#!/bin/bash

# Kill all child processes on exit
trap "kill 0" EXIT

# Set up the environment
# export CARLA_ROOT=~/Software/CARLA
export CARLA_ROOT=home/sorokin/Carla
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg:${CARLA_ROOT}/PythonAPI/carla/agents:${CARLA_ROOT}/PythonAPI/carla

# export SCENARIO_RUNNER_ROOT=${CARLA_ROOT}/ScenarioRunner
export SCENARIO_RUNNER_ROOT=/home/sorokin/scenario_runner

# Set up test parameters
WORLD_NAME=Town01
SCENARIO_NAME=FollowLeadingVehicle

# Start the simulator
${CARLA_ROOT}/CarlaUE4.exe \
    ${WORLD_NAME} \
    -quality-level=Low \
    -RenderOffScreen \
    &
sleep 10

# Load the world
python ${CARLA_ROOT}/PythonAPI/util/config.py --map ${WORLD_NAME}

# Load the scenario
LOG_ABSOLUTE_PATH=/tmp/carla/recordings
LOG_RELATIVE_PATH=$(realpath --relative-to=${SCENARIO_RUNNER_ROOT} ${LOG_ABSOLUTE_PATH})
python \
    ${SCENARIO_RUNNER_ROOT}/scenario_runner.py \
    --openscenario ${SCENARIO_RUNNER_ROOT}/srunner/examples/${SCENARIO_NAME}.xosc \
    --record ${LOG_RELATIVE_PATH} \
    &
sleep 10

# Load the controller
python \
    controller/manual.py

# Evaluate the metric
python \
    ${SCENARIO_RUNNER_ROOT}/metrics_manager.py \
    --metric ${SCENARIO_RUNNER_ROOT}/srunner/metrics/examples/distance_between_metrics.py \
    --log ${LOG_RELATIVE_PATH}/${SCENARIO_NAME}.log
