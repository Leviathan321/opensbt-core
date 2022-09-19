# Copyright (c) 2020 Universitat Autonoma de Barcelona (UAB)
# Copyright (c) 2022 fortiss GmbH
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import math

from srunner.metrics.tools.metrics_log import MetricsLog


class RawData:

    def evaluate(self, simulator, recording):
        info = simulator.get_client().show_recorder_file_info(recording, True)
        log = MetricsLog(info)

        ego_id = log.get_actor_ids_with_role_name("hero")[0]
        adv_id = log.get_actor_ids_with_role_name("adversary")[0]

        distance_profile = []
        frames_time_list = []

        ego_location_profile = []
        adv_location_profile = []

        ego_velocity_profile = []
        adv_velocity_profile = []

        ego_speed_profile = []
        adv_speed_profile = []

        ego_acceleration_profile = []
        adv_acceleration_profile = []

        ego_rotation_profile = []
        adv_rotation_profile = []

        ego_yaw_profile = []
        adv_yaw_profile = []

        ego_pitch_profile = []
        adv_pitch_profile = []

        ego_roll_profile = []
        adv_roll_profile = []

        start_ego, end_ego = log.get_actor_alive_frames(ego_id)
        start_adv, end_adv = log.get_actor_alive_frames(adv_id)

        start = max(start_ego, start_adv)
        end = min(end_ego, end_adv)

        collisions = log.get_actor_collisions(ego_id)

        simTime = log.get_elapsed_time(log.get_total_frame_count() - 1)

        for i in range(start, end):
            frames_time_list.append(log.get_elapsed_time(i))

            ego_location = log.get_actor_transform(ego_id, i).location
            adv_location = log.get_actor_transform(adv_id, i).location
            relative_position = ego_location - adv_location
            distance = math.sqrt(relative_position.x ** 2 + relative_position.y ** 2 + relative_position.z ** 2)
            ego_location_profile.append((ego_location.x, ego_location.y))
            adv_location_profile.append((adv_location.x, adv_location.y))
            distance_profile.append(distance)

            # Note: (pitch, yaw, roll), which corresponds to (Y-rotation,Z-rotation,X-rotation) in degrees
            ego_rotation = log.get_actor_transform(ego_id, i).rotation
            adv_rotation = log.get_actor_transform(adv_id, i).rotation
            ego_rotation_profile.append((ego_rotation.pitch, ego_rotation.yaw, ego_rotation.roll))
            adv_rotation_profile.append((adv_rotation.pitch, adv_rotation.yaw, adv_rotation.roll))
            ego_yaw_profile.append(ego_rotation.yaw)
            adv_yaw_profile.append(adv_rotation.yaw)
            ego_pitch_profile.append(ego_rotation.pitch)
            adv_pitch_profile.append(adv_rotation.pitch)
            ego_roll_profile.append(ego_rotation.roll)
            adv_roll_profile.append(adv_rotation.roll)

            ego_velocity = log.get_actor_velocity(ego_id, i)
            adv_velocity = log.get_actor_velocity(adv_id, i)
            ego_velocity_profile.append((ego_velocity.x, ego_velocity.y, ego_velocity.z))
            adv_velocity_profile.append([adv_velocity.x, ego_velocity.y, ego_velocity.z])

            # Note: speed is the magnitude of a velocity vector
            ego_speed = math.sqrt(ego_velocity.x ** 2 + ego_velocity.y ** 2 + ego_velocity.z ** 2)
            adv_speed = math.sqrt(adv_velocity.x ** 2 + adv_velocity.y ** 2 + adv_velocity.z ** 2)
            ego_speed_profile.append(ego_speed)
            adv_speed_profile.append(adv_speed)

            ego_acceleration = log.get_actor_acceleration(ego_id, i)
            adv_acceleration = log.get_actor_acceleration(adv_id, i)
            ego_acceleration_profile.append((ego_acceleration.x, ego_acceleration.y, ego_acceleration.z))
            adv_acceleration_profile.append((adv_acceleration.x, adv_acceleration.y, adv_acceleration.z))

        result = {
            "simTime": 0,
            "times": [],
            "location": {
                "ego": [],
                "adversary": []
            },
            "velocity": {
                "ego": [],
                "adversary": []
            },
            "speed": {
                "ego": [],
                "adversary": []
            },
            "acceleration": {
                "ego": [],
                "adversary": []
            },
            "yaw": {
                "ego": [],
                "adversary": []
            },
            "collisions": [],
            "actors": {},
            "otherParams": {}
        }

        result["simTime"] = simTime
        result["times"] = frames_time_list
        result["location"]["ego"] = ego_location_profile
        result["location"]["adversary"] = adv_location_profile
        result["velocity"]["ego"][0:2] = ego_velocity_profile
        result["velocity"]["adversary"] = adv_velocity_profile
        result["speed"]["ego"] = ego_speed_profile
        result["speed"]["adversary"] = adv_speed_profile
        result["acceleration"]["ego"] = ego_acceleration_profile
        result["acceleration"]["adversary"] = adv_acceleration_profile
        result["yaw"]["ego"] = ego_speed_profile
        result["yaw"]["adversary"] = adv_speed_profile
        result["collisions"] = collisions
        result["actors"] = {
            ego_id: "ego",
            adv_id: "adversary"
        }
        result["otherParams"]["distance"] = distance_profile

        return result