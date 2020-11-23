import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

SHOW_PREVIEW = True
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 100  # We need to on-ramp and drive. So increasing to 100


class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    reward = -300
    first_lane_change_on_freeway = True

    def __init__(self):
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(60.0)
        #self.world = self.client.get_world()
        #self.world = self.client.load_world('Town04')

        world_loaded_flag=False
        world_retry_count = 0
        while(not world_loaded_flag):
            try:
                self.world = self.client.load_world('Town04')
                world_loaded_flag = True
            except Exception:
                world_retry_count+=1
                time.sleep(20)
                if world_retry_count>10:
                    print("ERROR: Could not load world after 10 retries")
                    sys.exit(1)
                pass

        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []
        self.reward = -300
        self.first_lane_change_on_freeway = True

        desired_spawn_point = carla.Transform(carla.Location(
            131.7, -54.3, 9), carla.Rotation(0, 84, 0))
        
        car_loaded_flag=False
        while(not car_loaded_flag):
            try:
                self.vehicle = self.world.spawn_actor(self.model_3, desired_spawn_point)
                car_loaded_flag = True
            except Exception:
                time.sleep(20)
                pass

        #self.transform = random.choice(self.world.get_map().get_spawn_points())
        #self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(
            self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(
            colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        lanedetectsensor = self.blueprint_library.find("sensor.other.lane_invasion")  # or sensor.other.lane_invasion 
        self.lanedetectsensor = self.world.spawn_actor(lanedetectsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.lanedetectsensor)
        self.lanedetectsensor.listen(lambda event: self.lane_crossing(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        # print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action):

        on_ramp = False

        current_lane = self.vehicle.get_world().get_map().get_waypoint(self.vehicle.get_location())
        if current_lane.left_lane_marking.type == 'Solid' and current_lane.right_lane_marking.type == 'Solid':
            on_ramp = True


        if action == 0:  # full throttle left
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=1.0, steer=-0.75*self.STEER_AMT))
        elif action == 1:  # full throttle straight
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=1.0, steer=0))
        elif action == 2:  # full throttle right
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=1.0, steer=0.75*self.STEER_AMT))
        elif action == 3:  # half throttle left
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=0.5, steer=-0.5*self.STEER_AMT))
        elif action == 4:  # half throttle straight
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=0.5, steer=0))
        elif action == 5:  # half throttle right
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=0.5, steer=0.5*self.STEER_AMT))
        elif action == 6:  # full brake
            self.vehicle.apply_control(carla.VehicleControl(
                brake=1, steer=0))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        # Going from ramp to freeway
        new_lane = self.vehicle.get_world().get_map().get_waypoint(self.vehicle.get_location())
        if on_ramp == True and new_lane.left_lane_marking.type == 'Broken' and new_lane.right_lane_marking.type == 'Solid':
            on_ramp = False
            reward += 100
            first_lane_change_on_freeway = False
        

        done = False
        if len(self.lane_crossing) != 0:
            for x in self.lane_crossing:
                clm = x.crossed_lane_markings     #How many events in here?
                for marking in clm:
                    if marking == 'Solid' or marking == 'SolidSolid':
                        reward = -100
                        done = True
                    elif first_lane_change_on_freeway == False:  # Rewarding the first change on the freeway
                        reward += 100
                        first_lane_change_on_freeway = True

        if len(self.collision_hist) != 0:
            done = True
            reward = -200

        if done == False:
            if kmh < 50:
                if kmh > 40: # speed <40
                    reward -= 1
                elif kmh > 30:
                    reward -= 2
                else:
                    reward -= 3

            else:
                reward += 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None
