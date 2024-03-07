import logging as log
import os
import xml.etree.ElementTree as ET

def update_rel_xodr_path(xosc_filename, xodr_folder):
    tree = ET.parse(xosc_filename)
    root = tree.getroot()

    road_network = root.find(".//RoadNetwork")
    print(road_network)
    child = road_network.find("LogicFile")
    current_fpath = child.get("filepath")
    if os.path.isdir(current_fpath):
        # already absolute
        log.info("No update.")
        return
    
    text = f'''<Directory path="{xodr_folder + os.sep + current_fpath}"/>'''
    logic_file = ET.fromstring(text)
    # otherwise remove and create new one with absolute path
    if child is not None:
        road_network.remove(child)

    road_network.append(logic_file)
    tree.write(xosc_filename)

    log.info("xodr filepath set in OpenSCENARIO file.")

    return

