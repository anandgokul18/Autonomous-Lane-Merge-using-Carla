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
import random

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

    # Loading a Tesla Model 3 as the Ego car
    bp = blueprint_library.filter("model3")[0]

    # Choosing the spawn point for the ego car
    # Carla comes with ~200 spawn points.
    # List of spawn_points: = world.get_map().get_spawn_points()
    # random spawn_point: spawn_point = random.choice(world.get_map().get_spawn_points())
    # Choosing desired spawn point based on x,y and z in map
    # NOTE: Need to change this so that the car spawns on the on-ramp
    spawn_point = carla.Transform(carla.Location(
        205.1, -318.8, 0), carla.Rotation(0, -89, 0))

    # Spawining the Ego car actor
    vehicle = world.spawn_actor(bp, spawn_point)

    # Control the car.
    # For now, just go straight at full throttle
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

    # Adding our ego car to list of actors to cleanup
    actor_list.append(vehicle)

finally:
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")
