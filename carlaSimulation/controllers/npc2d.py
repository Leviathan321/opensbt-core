# Copyright (c) 2021 Universitat Autonoma de Barcelona (UAB)
# Copyright (c) 2022 fortiss GmbH
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import carla
import carlaSimulation.visualizations.simple as Simple

from agents.navigation.behavior_agent import BasicAgent

from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

MAX_SPEED = 3

class NpcAgent2d(AutonomousAgent):

    _agent = None
    _route_assigned = False
    _visual = None

    def __init__(self, simulator,max_speed = MAX_SPEED):
        super().__init__("")
        if not simulator.get_client().get_world().get_settings().no_rendering_mode:
            self._visual = Simple.start()

    def setup(self, _):
        self._agent = None

    def run_step(self, _, timestamp):
        if self._visual is not None:
            self._visual.update(timestamp)
        if not self._agent:
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    hero_actor = actor
                    break
            if hero_actor:
                self._agent = BasicAgent(hero_actor,MAX_SPEED)
            return carla.VehicleControl()
        else:
            return self._agent.run_step()

    def destroy(self):
        if self._visual is not None:
            self._visual.stop()
