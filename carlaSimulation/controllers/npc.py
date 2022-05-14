#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import carla

from agents.navigation.behavior_agent import BehaviorAgent

from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class NpcAgent(AutonomousAgent):

    _agent = None
    _route_assigned = False

    def setup(self, path_to_conf_file):
        self._agent = None

    def run_step(self, input_data, timestamp):
        if not self._agent:
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    hero_actor = actor
                    break
            if hero_actor:
                self._agent = BehaviorAgent(hero_actor, behavior='aggressive')
            return carla.VehicleControl()
        else:
            return self._agent.run_step()
