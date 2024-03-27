import pandas as pd
import json
from math import pi

class EsminiParser(object):

    @staticmethod
    def parse_log(filename: str):

        # Open the file for reading
        with open(filename, 'r') as file:
            # Read all lines into a list
            lines = file.readlines()

        # Remove the first 6 lines
        lines = lines[6:]

        # Open the file for writing and overwrite the contents
        with open(filename, 'w') as file:
            # Write the remaining lines back to the file
            file.writelines(lines)
            
        # read the csv file
        df = pd.read_csv(filename)
        out = {
                    "simTime": -1,
                    "times": [],
                    "location": {},
                    "velocity": {},
                    "speed": {},
                    "acceleration": {},
                    "yaw": {},
                    "collisions": [],
                    "actors" : {
                        "pedestrians" : [],
                        "vehicles" : []
                    },
                    "otherParams": {}
        }
        n_col_data_actor = 31
        n_col_start = 2
        n_actors =  int ((len(df.columns) - n_col_start - 1) / n_col_data_actor)

        # print(f"n_actors: {n_actors}")
        # print(df.iloc[0,2])
        # print(df.columns[1])
        # print(len(df.columns))
        actors = []
        out["times"] = df.iloc[:,1].to_list()

        map = {
            "name": 0,
            "speed": 2,
            "pos_x" : 11,
            "pos_y" : 12,
            "pos_z" : 13,
            "vel_x" : 14,
            "vel_y" : 15,
            "vel_z" : 16,
            "acc_x" : 17,
            "acc_y" : 18,
            "acc_z" : 19,
            "angle" : 24,
            "col_ids" : 30
            }


        for i in range(0,n_actors):
            def get_ind(entry_name):
                return n_col_start + i * n_col_data_actor +  map[entry_name]
            # actors
            actor = df.iloc[0, get_ind('name') ]
            actor = actor.replace(" ","")
            if actor == "ego" or actor == "hero":
                out["actors"][actor] = "ego"
            if actor == "adversary":
                out["actors"][actor] = "adversary"
            else:
                out["actors"]["vehicles"].append(actor)
            actors.append(actor)
            # speeds
            speeds = df.iloc[:, get_ind("speed") ].to_list()
            out["speed"][actor] = speeds

            # location
            pos_x = df.iloc[:, get_ind("pos_x") ].to_list()
            pos_y = df.iloc[:, get_ind("pos_y") ].to_list()
            pos = [e for e in zip(pos_x,pos_y)]
            out["location"][actor] = pos

            # velocity
            vel_x = df.iloc[:, get_ind("vel_x") ].to_list()
            vel_y = df.iloc[:, get_ind("vel_y") ].to_list()
            vel_z = df.iloc[:, get_ind("vel_z") ].to_list()
            vel = [e for e in zip(vel_x,vel_y, vel_z)]
            out["velocity"][actor] = vel

            # acceleration
            acc_x = df.iloc[:, get_ind("acc_x") ].to_list()
            acc_y = df.iloc[:, get_ind("acc_y") ].to_list()
            acc_z = df.iloc[:, get_ind("acc_z") ].to_list()
            acc = [e for e in zip(acc_x,acc_y, acc_z)]
            out["acceleration"][actor] = acc

            # angle
            angles = df.iloc[:, get_ind("angle") ].to_list()
            # convert to degree
            angles_new = [(180/pi)*a for a in angles]
            out["yaw"][actor] = angles_new

            # collision
            collisions = df.iloc[:, get_ind("col_ids") ].to_list()
            for c in collisions:
                if isinstance(c,str):
                    sclean = c.replace(" ","")
                if (isinstance(c,str) and len(sclean) > 0) or isinstance(c,int):
                    id = int(sclean)
                    col_tuple = (i,id)
                    if col_tuple not in out["collisions"] and \
                    col_tuple[::-1] not in out["collisions"]:
                        out["collisions"].append(col_tuple)
                        out["otherParams"]["isCollision"] = True
        return out