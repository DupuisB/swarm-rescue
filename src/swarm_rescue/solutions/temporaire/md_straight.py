"""
This program can be launched directly.
Example of how to control one drone
"""

import math
import os
import sys
from typing import List, Type, Tuple

import arcade
import numpy as np

# This line add, to sys.path, the path to parent path of this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spg_overlay.utils.path import Path
from spg_overlay.utils.pose import Pose
from spg_overlay.utils.utils import clamp, normalize_angle
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.gui_map.closed_playground import ClosedPlayground
from spg_overlay.gui_map.gui_sr import GuiSR
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.utils.misc_data import MiscData


class MyDroneStraight(DroneAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_position = np.array([150, 100])
        self.prev_diff_position = np.zeros(2)
        self.prev_diff_angle = 0

    def world_to_absolute(self, position):
        return np.array([position[0] + self.size_area[0] / 2, position[1] + self.size_area[1] / 2])

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def control_goto(self):
        """
        target_position is self.target_position
        Move the drone to a target position using PID, done when the drone is close enough (20 pixels)
        """
        diff_position = self.target_position - np.asarray(self.measured_gps_position())
        #self.measured_velocity() gives the velocity in both x and y directions
        if np.linalg.norm(diff_position) < 20 and np.linalg.norm(self.measured_velocity()) < 1:
            print("Arrived at target position")
            return {"forward": 0, "rotation": 0}

        desired_angle = math.atan2(diff_position[1], diff_position[0])

        diff_angle = normalize_angle(desired_angle - self.measured_compass_angle())
        deriv_diff_angle = normalize_angle(diff_angle - self.prev_diff_angle)

        # PID controller for rotation
        Ku_angle = 11.16
        Tu_angle = 2.0
        Kp_angle = 0.8 * Ku_angle
        Kd_angle = Ku_angle * Tu_angle / 40.0
        rotation = Kp_angle * diff_angle + Kd_angle * deriv_diff_angle
        rotation = clamp(rotation, -1.0, 1.0)

        deriv_diff_position = diff_position - self.prev_diff_position

        # PID controller for forward movement
        Ku_position = 25 / 100
        Tu_position = 26
        Kp_position = 0.8 * Ku_position
        Kd_position = Ku_position * Tu_position / 10.0
        print(f"forward = {Kp_position * diff_position[0]} + {Kd_position * deriv_diff_position[0]}")
        forward = Kp_position * diff_position[0] + Kd_position * deriv_diff_position[0]
        forward = clamp(forward, -1.0, 1.0)

        self.prev_diff_position = diff_position
        self.prev_diff_angle = diff_angle
        return {"forward": forward, "rotation": rotation}

    def control(self):
        command = self.control_goto()
        return command


    def draw_bottom_layer(self):
        # Draw a line to the target position
        x1, y1 = self.world_to_absolute(self.true_position())
        x2, y2 = self.world_to_absolute(self.target_position)
        arcade.draw_line(x1, y1, x2, y2,
                         arcade.color.RED, 2)
        arcade.draw_circle_filled(x2, y2, 5, arcade.color.RED)

class MyMapRandom(MapAbstract):
    def __init__(self):
        super().__init__()

        # PARAMETERS MAP
        self._size_area = (400, 400)

        # POSITIONS OF THE DRONES
        self._number_drones = 1
        self._drones_pos = [((0, 0), 0)]

        self._drones: List[DroneAbstract] = []

    def construct_playground(self, drone_type: Type[DroneAbstract]):
        playground = ClosedPlayground(size=self._size_area)

        # POSITIONS OF THE DRONES
        misc_data = MiscData(size_area=self._size_area,
                             number_drones=self._number_drones)
        for i in range(self._number_drones):
            drone = drone_type(identifier=i, misc_data=misc_data)
            self._drones.append(drone)
            playground.add(drone, self._drones_pos[i])

        return playground


def main():
    my_map = MyMapRandom()

    playground = my_map.construct_playground(drone_type=MyDroneStraight)

    gui = GuiSR(playground=playground,
                the_map=my_map,
                use_keyboard=False,
                use_mouse_measure=True,
                enable_visu_noises=False,
                )

    gui.run()


if __name__ == '__main__':
    main()
