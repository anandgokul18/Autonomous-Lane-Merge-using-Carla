#! /usr/bin/env python

"""

Welcome to Autonomous Lane Merging using CARLA Simulator

This work is part of the 'Autonomous Lane Merging using Reinforcement Learning'
project, authored by Anand Gokul (Github: @anandgokul18) and Ashwini Raja (Github: ashwini-raja)

License: MIT Open License (https://opensource.org/licenses/MIT)
Year: 2020
Carla Version: 0.9.5 (Code may need changes in future versions of CARLA. Provided as is, no warranties provided :x)

"""

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# These will be the actors whcih will be needed for the list. Used for both init and cleanup
actor_list = []

try:
    # We need to ensure the delay factors if using non-local servers, say on another machine dedicated to training.
    # For non-local machines, delay= sending+training every iter + receivin, all cause delays... use local if possible
    # NOTE TO SELF: Once final code is set in stone, revisit this and see the delay for the remote machine
    client = carla.Client("localhost", 2000)

    # Setting 2 sec timeout (the docs use 10)
    client.set_timeout(2.0)

    # We are loading the World as Town04 since Town04 map is the highway which we are planning to train on
    # Default: world = client.get_world()
    world = client.load_world('Town04')

    blueprint_library = world.get_blueprint_library()

finally:
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")
