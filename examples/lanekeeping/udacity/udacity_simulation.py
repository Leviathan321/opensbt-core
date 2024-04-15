# imports related to OpenSBT
from pathlib import Path
import time
from matplotlib import pyplot as plt
import pymoo
from examples.lanekeeping.agent.agent_utils import calc_yaw_ego
from opensbt.simulation.simulator import Simulator, SimulationOutput

from opensbt.model_ga.individual import IndividualSimulated
from examples.lanekeeping.udacity.os_utils import kill_udacity_simulator

pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

# all other imports
from distutils import config
from typing import List, Dict, Tuple
import numpy as np
import os
import gym
import cv2
from numpy import uint8
from tensorflow.keras.models import load_model
import requests

# related to this simulator
from examples.lanekeeping.road_generator.road_generator import RoadGenerator
from examples.lanekeeping.udacity.env.udacity_gym_env import (
    UdacityGymEnv_RoadGen
)

from examples.lanekeeping.road_generator.custom_road_generator import CustomRoadGenerator

import examples.lanekeeping.config as config

from examples.lanekeeping.agent.dnn_agent import DnnAgent
from examples.lanekeeping.self_driving.supervised_agent import SupervisedAgent
from timeit import default_timer as timer
import logging as log

class UdacitySimulator(Simulator):
    # initial_pos = (125, 1.90000000, 1.8575, 8)
    initial_pos=(125.0, 0, -28.0, 8)
    
    @staticmethod
    def simulate(
        list_individuals: List[IndividualSimulated],
        variable_names: List[str],
        scenario_path: str,
        sim_time: float,
        time_step: float,
        do_visualize: bool = False,
    ) -> List[SimulationOutput]:
        """
        Runs all indicidual simulations and returns simulation outputs
        """
        results = []
        test_generator = CustomRoadGenerator(map_size=250,
                                            num_control_nodes=len(list_individuals[0]),
                                            seg_length=config.SEG_LENGTH)
        file_path = "/home/sorokin/Projects/testing/Multi-Simulation/opensbt-multisim/road_generator/roads_matteo_maxibon.txt"
        env = None
        # obs, done, info = env.observe()
        agent = SupervisedAgent(
                        env_name=config.UDACITY_SIM_NAME,
                        model_path=config.DNN_MODEL_PATH,
                        min_speed=config.MIN_SPEED,
                        max_speed=config.MAX_SPEED,
                        input_shape=config.INPUT_SHAPE,
                        predict_throttle=False,
                        fake_images=False
                        )
        #print("[UdacitySimulator] loaded model")
        
        for ind in list_individuals:
            speed = 0
            try:
                speeds = []
                pos = []
                xte = []
                steerings = []
                throttles = []

                instance_values = [v for v in zip(variable_names, ind)]
                angles = UdacitySimulator._process_simulation_vars(instance_values)
                road = test_generator.generate(
                                starting_pos=UdacitySimulator.initial_pos,
                                angles=angles,
                                simulator_name=config.UDACITY_SIM_NAME)
                # road = test_generator.generate()
                waypoints = road.get_string_repr()
                
                # set up of params
                done = False

                if env is None:
                    env = UdacityGymEnv_RoadGen(
                        seed=1,
                        test_generator=test_generator,
                        exe_path=config.UDACITY_EXE_PATH)
                    
                obs = env.reset(skip_generation=False, track_string=waypoints)
         
                start = timer()
                
                fps_time_start = time.time()
                counter = 0
                counter_all = []

                while not done:
                    # calculate fps
                    if time.time() - fps_time_start > 1:
                        #reset 
                        log.info(f"Frames in 1s: {counter}")
                        log.info(f"Time passed: {time.time() - fps_time_start}")
                        
                        counter_all.append(counter)
                        counter = 0
                        fps_time_start = time.time()
                    else:
                        counter += 1
                    # time.sleep(0.15)
                    actions = agent.predict(obs=obs, 
                                state = dict(speed=speed, 
                                            simulator_name=config.UDACITY_SIM_NAME)
                    )    
                    # # clip action to avoid out of bound errors
                    if isinstance(env.action_space, gym.spaces.Box):
                        actions = np.clip(
                            actions, 
                            env.action_space.low, 
                            env.action_space.high
                        )
                    # obs is the image, info contains the road and the position of the car
                    obs, done, info = env.step(actions)

                    speed = 0.0 if info.get("speed", None) is None else info.get("speed")

                    speeds.append(info["speed"])
                    pos.append(info["pos"])
                    
                    if config.CAP_XTE:
                        xte.append(info["cte"] 
                                        if abs(info["cte"]) <= config.MAX_XTE \
                                        else config.MAX_XTE)
                        
                        assert np.all(abs(np.asarray(xte)) <= config.MAX_XTE), f"At least one element is not smaller than {config.MAX_XTE}"
                    else:
                        xte.append(info["cte"])
                    steerings.append(actions[0][0])
                    throttles.append(actions[0][1])

                    end = timer()
                    time_elapsed = int(end - start)
                    if time_elapsed % 2 == 0:
                        pass#print(f"time_elapsed: {time_elapsed}")
                    elif time_elapsed > config.TIME_LIMIT:  
                        #print(f"Over time limit, terminating.")    
                        done = True       
                    elif abs(info["cte"]) > config.MAX_XTE:
                        #print("Is above MAXIMAL_XTE. Terminating.")
                        done = True
                    else:
                        pass
                
                fps_rate = np.sum(counter_all)/time_elapsed
                log.info(f"FPS rate: {fps_rate}")

                # morph values into SimulationOutput Object
                result = SimulationOutput(
                    simTime=time_elapsed,
                    times=[x for x in range(len(speeds))],
                    location={
                        "ego": [(x[0], x[1]) for x in pos],  # cut out z value
                    },
                    velocity={
                        "ego": UdacitySimulator._calculate_velocities(pos, speeds),
                    },
                    speed={
                        "ego": speeds,
                    },
                    acceleration={"ego": UdacitySimulator.calc_acceleration(speeds=speeds, fps=20)},
                    yaw={
                        "ego" : calc_yaw_ego(pos)
                    },                   
                    collisions=[],
                    actors={
                        "ego" : "ego",
                        "pedestrians" : [],
                        "vehicles": ["ego"]
                    },
                    otherParams={"xte": xte,
                                "simulator" : "Udacity",
                                "road": road.get_concrete_representation(to_plot=True),
                                "steerings" : steerings,
                                "throttles" : throttles,
                                "fps_rate": fps_rate}
                )

                results.append(result)
            except Exception as e:
                #print(f"Received exception during simulation {e}")

                raise e
            finally:
                if env is not None:
                    env.close()
                    env = None
                kill_udacity_simulator()
                #print("Finished individual")
        # # close the environment
        # env.close()
        return results

    @staticmethod
    def _calculate_velocities(positions, speeds) -> Tuple[float, float, float]:
        """
        Calculate velocities given a list of positions and corresponding speeds.
        """
        velocities = []
        for i in range(len(positions) - 1):
            displacement = np.array(positions[i + 1]) - np.array(positions[i])
            direction = displacement / np.linalg.norm(displacement)
            velocity = direction * speeds[i]
            velocities.append(velocity)

        return velocities

    @staticmethod
    def _process_simulation_vars(
        instance_values: List[Tuple[str, float]],
    ) -> Tuple[List[int]]:
        angles = []
        for i in range(0, len(instance_values)):
            new_angle = int(instance_values[i][1])
            angles.append(new_angle)

        return angles
    
    @staticmethod
    def calc_acceleration(speeds: List, fps: int):
        acc=[0]
        for i in range(1,len(speeds)):
            a = (speeds[i] - speeds[i-1])*fps / 3.6 # convert to m/s
            acc.append(a)
        return acc
