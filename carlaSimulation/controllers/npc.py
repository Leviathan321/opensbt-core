#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import print_function

import pygame
import carla

from agents.navigation.behavior_agent import BehaviorAgent

from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class HumanInterface(object):

    def __init__(self):
        self._width = 300
        self._height = 200
        self._surface = None

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode((self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Agent")

    def run_interface(self, input_data):
        image_left = input_data['Left'][1][:, :, -2::-1]
        self._surface = pygame.surfarray.make_surface(image_left.swapaxes(0, 1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()

    def quit_interface(self):
        pygame.quit()


class NpcAgent(AutonomousAgent):

    _agent = None
    _route_assigned = False

    def setup(self, path_to_conf_file):
        self._agent = None
        self._hic = HumanInterface()

    def sensors(self):
        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},
        ]
        return sensors

    def run_step(self, input_data, timestamp):
        self._hic.run_interface(input_data)

        if not self._agent:

            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    hero_actor = actor
                    break

            if not hero_actor:
                return carla.VehicleControl()

            self._agent = BehaviorAgent(hero_actor, behavior='aggressive')

            return carla.VehicleControl()

        else:
            return self._agent.run_step()

    def destroy(self):
        self._hic.quit_interface = True
