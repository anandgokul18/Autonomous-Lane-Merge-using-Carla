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

import random
import time
import numpy as np
import cv2  # opencv-python


IM_WIDTH = 640
IM_HEIGHT = 480


def process_img(image):
    # convert the raw_data to an array
    i = np.array(image.raw_data)
    # was flattened, so we're going to shape it.
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    # remove the alpha (basically, remove the 4th index  of every pixel. Converting RGBA to RGB)
    i3 = i2[:, :, :3]
    # show it.
    cv2.imshow("", i3)
    cv2.waitKey(1)
    # normalize
    return i3/255.0


# These will be the actors whcih will be needed for the list. Used for both init and cleanup
actor_list = []

try:
    # We need to ensure the delay factors if using non-local servers, say on another machine dedicated to training.
    # For non-local machines, delay= sending+training every iter + receivin, all cause delays... use local if possible
    # NOTE TO SELF: Once final code is set in stone, revisit this and see the delay for the remote machine
    client = carla.Client("127.0.0.1", 2000)

    # Setting 2 sec timeout (the docs use 10)
    client.set_timeout(60.0)

    # We are loading the World as Town04 since Town04 map is the highway which we are planning to train on
    # Default: world = client.get_world()
    world = client.load_world('Town04')

    blueprint_library = world.get_blueprint_library()

    # Loading a Tesla Model 3 as the Ego car
    bp = blueprint_library.filter("model3")[0]

    # Choosing the spawn point for the ego car
    # Carla comes with ~200 spawn points.
    # List of spawn_points: = world.get_map().get_spawn_points()
    # random spawn_point: 
    #spawn_point = random.choice(world.get_map().get_spawn_points())
    # Choosing desired spawn point based on x,y and z in map
    # NOTE: Need to change this so that the car spawns on the on-ramp
    
    spawn_point1 = carla.Transform(carla.Location(206.7, -357.4, 1), carla.Rotation(0, -86, 0)) #Type=Driving
    spawn_point2 = carla.Transform(carla.Location(211.7, -363.8, 1), carla.Rotation(0, -1, 0)) #Type=Driving

    spawn_point3 = carla.Transform(carla.Location(131.7, -54.3, 9), carla.Rotation(0, 84, 0))
    spawn_point4 = carla.Transform(carla.Location(103.7, -1.5, 12), carla.Rotation(0, 148, 0))

    # Choosing one of the 2 on-ramp spawn points randomly
    spawn_point = random.choice([spawn_point1, spawn_point2, spawn_point3, spawn_point4])

    # Spawining the Ego car actor
    vehicle = world.spawn_actor(bp, spawn_point)

    # Control the car.
    # For now, just go straight at full throttle
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))

    # Adding our ego car to list of actors to cleanup
    actor_list.append(vehicle)

    # Adding Camera Sensor
    cam_bp = blueprint_library.find('sensor.camera.rgb')

    # change the dimensions of the image and field of view to be specific for the neural network
    cam_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
    cam_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
    cam_bp.set_attribute('fov', '110')

    # Adjust sensor relative to the ego vehicle. x is forward, y is left/right, z is up/down
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)

    # add sensor to list of actors
    actor_list.append(sensor)

    # Get the actual data from the sensor and preprocess the data using lambda function
    sensor.listen(lambda data: process_img(data))

    #Print the current waypoint type
    waypoint = world.get_map().get_waypoint(vehicle.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.OnRamp))
    print("Current lane type: " + str(waypoint.lane_type))
    # Check current lane change allowed
    print("Current Lane change:  " + str(waypoint.lane_change))
    # Left and Right lane markings
    print("L lane marking type: " + str(waypoint.left_lane_marking.type))
    print("L lane marking change: " + str(waypoint.left_lane_marking.lane_change))
    print("R lane marking type: " + str(waypoint.right_lane_marking.type))
    print("R lane marking change: " + str(waypoint.right_lane_marking.lane_change))

    # 20 sec training
    time.sleep(20)

finally:
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")
