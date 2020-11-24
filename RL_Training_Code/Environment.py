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

from skimage.color import rgb2gray

SHOW_PREVIEW = True
IM_WIDTH = 300 #640
IM_HEIGHT = 240 #480
SECONDS_PER_EPISODE = 30  # We need to on-ramp and drive. So increasing to 100


class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    first_lane_change_on_freeway = True
    lane_crossings = []

    def __init__(self):
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(60.0)
        
        # FIRST do client.load_world(), then, do get_world()
        self.world = self.client.get_world()
        #self.world = self.client.load_world('Town04')

        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []
        self.first_lane_change_on_freeway = True
        self.lane_crossings = []

        desired_spawn_point = carla.Transform(carla.Location(-33.3, -87.5, 2), carla.Rotation(0, 44, 0))

        #desired_spawn_point = carla.Transform(carla.Location(-25.2, -78.7, 1), carla.Rotation(0, 57, 0))

        self.vehicle = self.world.spawn_actor(self.model_3, desired_spawn_point)

        #self.transform = random.choice(self.world.get_map().get_spawn_points())
        #self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"75") #Sentdex: 110

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(
            self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=0.0, reverse=False, hand_brake=False))
        time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(
            colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        lanedetectsensor = self.blueprint_library.find(
            "sensor.other.lane_invasion")  # or sensor.other.lane_invasion
        self.lanedetectsensor = self.world.spawn_actor(
            lanedetectsensor, transform, attach_to=self.vehicle)
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

    def lane_crossing(self,event):
        self.lane_crossings.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        # print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]

        #grayscale = rgb2gray(i3)

        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action):

        on_ramp = False

        # Starting reward for current step.
        reward = 0

        # Survival reward
        reward += 1 #+1 for each survival step

        current_lane = self.vehicle.get_world().get_map().get_waypoint(self.vehicle.get_location())
        if str(current_lane.left_lane_marking.type) == 'Solid' and str(current_lane.right_lane_marking.type) == 'Solid':
            print("[LOG] On-ramp is true")
            on_ramp = True

        if action == 0:  # full throttle left
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=1.0, steer=-0.25*self.STEER_AMT, reverse= False, hand_brake=False))
            #print("[LOG] Action 0")
        elif action == 1:  # full throttle straight
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=1.0, steer=0, reverse= False, hand_brake=False))
            #print("[LOG] Action 1")
        elif action == 2:  # full throttle right
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=1.0, steer=0.25*self.STEER_AMT, reverse= False, hand_brake=False))
            #print("[LOG] Action 2")
        elif action == 3:  # half throttle straight
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=0.5, steer=0, reverse= False, hand_brake=False))
            #print("[LOG] Action 3")
        elif action == 4:  # crawl straight
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=0.1, steer=0, reverse= False, hand_brake=False))
            #print("[LOG] Action 4")

        

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        done = False

        # Going from ramp to freeway
        new_lane = self.vehicle.get_world().get_map().get_waypoint(self.vehicle.get_location())
        if on_ramp == True and str(new_lane.left_lane_marking.type) == 'Broken' and str(new_lane.right_lane_marking.type) == 'Solid':
            on_ramp = False
            reward += 800
            self.first_lane_change_on_freeway = False
            print("[LOG] Ramp to freeway!")
            done = True ### Phase 1 training
        
        
        # TODO NOTE Need to optimize this code so that we dont iterate over entire list everytime!
        if len(self.lane_crossings) != 0:
            for x in self.lane_crossings:
                clm = x.crossed_lane_markings  # How many events in here?
                for marking in clm:
                    #print(str(marking.type)) 
                    if str(marking.type) != 'Broken': #str(marking.type) == 'Solid' or str(marking.type) == 'SolidSolid' or str(marking.type) == 'Curb' or str(marking.type) == 'Other':
                        reward += -50
                        print(f"[LOG] {str(marking.type)} Crossed...Penalty")
                        done = False
                    elif self.first_lane_change_on_freeway == False and str(marking.type) == 'Broken':  # Rewarding the first change on the freeway
                        reward += 100
                        self.first_lane_change_on_freeway = True
                        print("[LOG] First lane change on freeway")
                    else: # Penalizing unnecessary lane changes
                        reward += -10
                        print("[LOG] Lane Change penalty")

        if len(self.collision_hist) != 0:
            done = True
            print("[LOG] Collided...Done")
            reward += -200

        if done == False:
            if kmh < 50:
                #Phase 1...No speed incentive
                #reward += -1
                pass
            elif kmh >= 50:
                reward += 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None
