import math

from srunner.metrics.tools.metrics_log import MetricsLog

class DistanceBetweenVehicles:

    def evaluate(self, simulator, recording):
        info = simulator.get_client().show_recorder_file_info(recording, True)
        log = MetricsLog(info)

        ego_id = log.get_actor_ids_with_role_name("hero")[0]
        adv_id = log.get_actor_ids_with_role_name("adversary")[0]

        dist_list = []
        frames_list = []

        start_ego, end_ego = log.get_actor_alive_frames(ego_id)
        start_adv, end_adv = log.get_actor_alive_frames(adv_id)
        start = max(start_ego, start_adv)
        end = min(end_ego, end_adv)

        for i in range(start, end):
            ego_location = log.get_actor_transform(ego_id, i).location
            adv_location = log.get_actor_transform(adv_id, i).location

            if adv_location.z < -10:
                continue

            dist_v = ego_location - adv_location
            dist = math.sqrt(dist_v.x * dist_v.x + dist_v.y * dist_v.y + dist_v.z * dist_v.z)

            dist_list.append(dist)
            frames_list.append(i)

        return (frames_list, dist_list)
    