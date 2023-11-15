import math
from copy import deepcopy
from typing import Optional
import matplotlib.pyplot as plt
from queue import Queue
from enum import Enum
import arcade

import numpy as np

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle, sign, bresenham, clamp
from spg_overlay.entities.rescue_center import RescueCenter, wounded_rescue_center_collision
from spg_overlay.entities.wounded_person import WoundedPerson
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor

class MyDroneLeftWall2(DroneAbstract):
    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)
        self.x, self.y = None, None
        self.etape = 1
        self.prev_error = 0
        self.integral = 0

    def process_lidar_sensor(self):
        """
        Makes sure closest wall is at 90 degrees
        Makes sur the distance to the wall is between 40 and 60 pixels
        Rotation is counter-clockwise
        Moves diagonally (more speed)
        """
        # Get the angle of the closest wall
        closest_wall_number = None
        # Get the distance to the closest wall
        closest_wall_distance = float('inf')
        rotation, lateral, forward = 0, 0, 0
        for i in range(len(self.lidar_values())):
            if 25 <= i <= 110:
                continue
            dist = self.lidar_values()[i]
            if dist < closest_wall_distance:
                closest_wall_number = i
                closest_wall_distance = dist
        if closest_wall_number < 67:
            closest_wall_number += 180

        # Simple PI controller to follow the wall
        Kp = 0.5
        Ki = 0.1
        error = 40 - closest_wall_distance
        self.integral += error
        control_signal = Kp * error + Ki * (error - self.prev_error)
        rotation = control_signal
        print(f'rotation: {rotation}')
        rotation = clamp(rotation, -1.0, 1.0)
        forward = 1 - Ki * self.integral
        forward = clamp(forward, -1.0, 1.0)
        lateral = forward
        return rotation, lateral, forward

    def process_semantic_sensor(self):
        pass

    def control(self):
        """Makes the drone follow the right wall"""
        # Get the coordinates of the drone
        x, y = self.measured_gps_position()
        angle = self.measured_compass_angle()  # in radians

        forward, lateral, rotation, grasper = 0, 0, 0, 0

        # Command:
        rotation, lateral, forward = self.process_lidar_sensor()

        command = {
            'forward': forward,
            'lateral': lateral,
            'rotation': rotation,
            'grasper': grasper
        }
        self.etape += 1

        return command

    def define_message_for_all(self):
        """
        Define the message, the drone will send to and receive from other surrounding drones.
        """
        pass