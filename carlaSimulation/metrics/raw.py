import math
from simulation.simulator import SimulationOutput

from srunner.metrics.tools.metrics_log import MetricsLog

# Return raw data 
class RawData:

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
        
        #collisions = log.get_collisions(ego_id)

        for i in range(start, end):
            frames_list.append(i)

            ego_location = log.get_actor_transform(ego_id, i).location
            adv_location = log.get_actor_transform(adv_id, i).location

            dist_v = ego_location - adv_location

            dist = math.sqrt(dist_v.x * dist_v.x + dist_v.y * dist_v.y + dist_v.z * dist_v.z)
            dist_list.append(dist)
       
            ego_speed = log.get_actor_velocity(ego_id,i)
            adv_speed = log.get_actor_velocity(adv_id,i)
        

        return (frames_list, ego_location, adv_location, adv_speed, ego_speed, dist_list)
    
