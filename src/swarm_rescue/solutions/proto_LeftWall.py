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

class MyDroneLeftWall(DroneAbstract):
    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)
        self.x, self.y = 0, 0
        self.etape = 1
        self.isGrabing = False
        self.idle = 0
        self.idleRotation = False

    def process_lidar_sensor(self):
        """
        Makes sure closest wall is at 90 degrees
        Makes sur the distance to the wall is between 40 and 60 pixels
        Rotation is counter-clockwise
        Moves diagonally (more speed)
        """
        rotation, lateral, forward = 0, 0, 0

        # Finds closest wall that is approximately to the left
        closest_wall_number = None
        closest_wall_distance = float('inf')
        for i in range(len(self.lidar_values())):
            if 25 <= i <= 110:
                continue
            dist = self.lidar_values()[i]
            if dist < closest_wall_distance:
                closest_wall_number = i
                closest_wall_distance = dist

        if closest_wall_number < 67: #Normalize angle
            closest_wall_number += 180
        best_turn_angle = closest_wall_number - 157
        # Simple P controller for rotation
        rotation = best_turn_angle * 0.007
        if closest_wall_distance < 20: #Too close
            forward, lateral = 1, 0
            rotation *= 3
        elif closest_wall_distance < 30: #Perfect
            forward, lateral = 1, 1
        elif closest_wall_distance < 40: #Too far
            forward, lateral = 0, 1
            rotation *= 3
        else: #Way too far
            forward, lateral = 0, 1
            rotation *= 7

        rotation = clamp(rotation, -1.0, 1.0)

        #Turn faster if wall in front
        if self.lidar_values()[112] < 80:
            rotation = -1.0
            lateral = -0.5

        #Turn faster if it sees a gap (wall stops)
        for i in range(135, 160):
            if abs(self.lidar_values()[i] - self.lidar_values()[i+1]) > 20 and abs(closest_wall_number - 157.5) < 5:
                print("Wall stops")
                rotation = 1.0
                forward = 0

        return rotation, lateral, forward

    def process_semantic_sensor(self):
        """
        If not grabbing, go to closest wounded person
        If grabbing, go to closest rescue center
        """
        found_objective = False
        forward, rotation, lateral = 0, 0, 0
        for i in range(len(self.semantic_values())):
            data = self.semantic_values()[i]
            if not self.isGrabing: #Go to closest wounded person
                if 0 < data.distance < 30 and data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                    self.isGrabing = True #Grab if close enough
                elif data.distance < 100 and data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                    found_objective = True
            else: #Go to closest rescue center
                if 0 < data.distance < 20 and data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    self.isGrabing = False #Drop if close enough
                elif data.distance < 100 and data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    found_objective = True

            rotation = clamp(data.angle, -1.0, 1.0)
            if abs(data.angle) < 0.2:
                forward = 1
            if found_objective:
                break

        return found_objective, rotation, lateral, forward

    def follow_direction(self, angle, goal_direction):
        pass

    def control(self):
        """Makes the drone follow the left wall"""
        forward, lateral, rotation, grasper = 0, 0, 0, 0

        # Command:
        found_objective, rotation, lateral, forward = self.process_semantic_sensor()
        if not found_objective: #If no objective, follow wall
            rotation, lateral, forward = self.process_lidar_sensor()
        if self.isGrabing:
            grasper = 1

        if self.idle > 20: #If stuck for long enough, set idleRotation to True
            self.idleRotation = True
        if self.idleRotation: #If idleRotation, rotate
            rotation, lateral, forward = -1, 0, 0
            self.idle -= 1
        elif self.odometer_values()[0] < 1: #If not moving, add to stuck counter
            self.idle += 1
        if self.idle == 0: #Rotation done -> reset idleRotation
            self.idleRotation = False

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