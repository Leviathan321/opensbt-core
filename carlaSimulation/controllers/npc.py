# Copyright (c) 2021 Universitat Autonoma de Barcelona (UAB)
# Copyright (c) 2022 fortiss GmbH
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import carla

from carlaSimulation.visualizations.real import CameraView

from agents.navigation.behavior_agent import BehaviorAgent
#from agents.navigation.behavior_agent import BasicAgent

from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

MAX_SPEED = 3

class NpcAgent(AutonomousAgent):

    _agent = None
    _route_assigned = False
    _visual = None

    def __init__(self, simulator,max_speed = MAX_SPEED):
        super().__init__("")
        if not simulator.get_client().get_world().get_settings().no_rendering_mode:
            self._visual = CameraView('center')

    def setup(self, _):
        self._agent = None

    def run_step(self, input_data, _):
        if self._visual is not None:
            self._visual.run(input_data)
        if not self._agent:
            hero_actor = None

            for actor in CarlaDataProvider.get_world().get_actors():            
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    print(actor.attributes)
                    hero_actor = actor
                    break
            if hero_actor:
                self._agent = BehaviorAgent(hero_actor, behavior='aggressive')
                #self._agent = BasicAgent(hero_actor,MAX_SPEED)
                self._agent.follow_speed_limits = False
            return carla.VehicleControl()
        else:
            self._agent.follow_speed_limits = False
            return self._agent.run_step()

    def sensors(self):
        sensors = []
        if self._visual is not None:
            sensors.append(
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.7, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 800, 'height': 600, 'fov': 100,
                    'id': 'center'
                }
            )
        return sensors

    def destroy(self):
        if self._visual is not None:
            self._visual.quit = True
