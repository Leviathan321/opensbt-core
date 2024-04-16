import os
import tensorflow as tf

import gym
import numpy as np
from tensorflow.keras.models import load_model

from examples.lanekeeping.agent.agent import Agent
from examples.lanekeeping.config import DEFAULT_THROTTLE,\
                MAX_SPEED, MIN_SPEED, IN_HEIGHT, IN_WIDTH, \
                DONKEY_SIM_NAME, UDACITY_SIM_NAME 
        
from examples.lanekeeping.self_driving.utils.dataset_utils import preprocess
import time

class DnnAgent(Agent):

    def __init__(
            self,
            env: gym.Env,
            model_path: str,
            max_speed: int = MAX_SPEED,
            min_speed: int = MIN_SPEED,
            predict_throttle: bool = False,
    ):
        super().__init__(env=env)

        assert os.path.exists(model_path), 'Model path {} not found'.format(model_path)
        with tf.device('cpu:0'):
            self.model = load_model(filepath=model_path,
                        compile=False)

        self.predict_throttle = predict_throttle
        self.model_path = model_path

        self.max_speed = max_speed
        self.min_speed = min_speed

        self.model.compile(loss="sgd", metrics=["mse"])

        self.counter = 0

    def predict(self, 
                obs: np.ndarray, 
                speed: float = 0.0, 
                simulator_name: str = None) -> np.ndarray:
    
        #obs = preprocess(image=obs, simulator_name=simulator_name)
        # if simulator_name == UDACITY_SIM_NAME:
        #     obs = crop(image=obs, simulator_name=simulator_name)
        #     obs = resize(image=obs, width=IN_WIDTH, height=IN_HEIGHT)
        #     obs = bgr2yuv(image=obs)
            
        #     folder = HEATMAP_OUTPUT_UDACITY

        # elif simulator_name == DONKEY_SIM_NAME:
        #     obs = crop(image=obs, simulator_name=simulator_name)
        #     obs = resize(image=obs, width=IN_WIDTH, height=IN_HEIGHT)
        #     obs = bgr2yuv(image=obs)

        folder = HEATMAP_OUTPUT_DONKEY if simulator_name == DONKEY_SIM_NAME else HEATMAP_OUTPUT_UDACITY

        preprocess(env_name=simulator_name,
                   image=obs, 
                   fake_images=False)

        # the model expects 4D array
        obs = np.array([obs])

        WRITE_HEATMAP = False

        if WRITE_HEATMAP:
            interval = 3
            if self.counter == 0:
                # remove old images
                import glob
                files = glob.glob(folder + "/*")
                for f in files:
                    os.remove(f)

            self.counter = self.counter + 1 

            if self.counter % interval == 0:
                # save image
                import cv2
                cv2.imwrite(f"{folder}/image_{self.counter}.jpg", obs)

        multiplier = 0.1

        if self.predict_throttle:
            action = self.model.predict(obs, batch_size=1, verbose=0)
            steering, throttle = action[0], action[1]
        else:
            steering = float(self.model.predict(obs, batch_size=1, verbose=0))
            if simulator_name == UDACITY_SIM_NAME:
                steering = 0.2 * steering
            print(f"CURRENT SPEED: {speed}")

            if speed > self.max_speed:
                print("SLOWING DOWN")
                speed_limit = self.min_speed  # slow down
            else:
                speed_limit = self.max_speed

            # steering = self.change_steering(steering=steering)
            throttle = multiplier * np.clip(a=1.0 - steering ** 2 - (speed / speed_limit) ** 2, a_min=0.0, a_max=1.0)
            print(f"computed throttle: {throttle}")
        #override throttle (testing) 
        #throttle = DEFAULT_THROTTLE
        return np.asarray([[steering, throttle]], dtype = np.float32)