"""
Show the lidar data as a point cloud in a separate window
"""
"""
This program can be launched directly.
To move the drone, you have to click on the map, then use the arrows on the keyboard
"""

import os
import sys
from typing import Type

import cv2
import numpy as np

from spg.playground import Playground
from spg.utils.definitions import CollisionTypes

# This line add, to sys.path, the path to parent path of this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from maps.walls_medium_02 import add_walls, add_boxes
from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR
from spg_overlay.utils.grid import Grid
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.pose import Pose
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.rescue_center import RescueCenter, wounded_rescue_center_collision
from spg_overlay.gui_map.closed_playground import ClosedPlayground
from spg_overlay.gui_map.gui_sr import GuiSR
from spg_overlay.gui_map.map_abstract import MapAbstract


class LidarVisualization(Grid):
    """Simple visualization of the lidar rays"""

    def __init__(self,
                 size_area_world,
                 resolution: float,
                 lidar):
        super().__init__(size_area_world=size_area_world, resolution=resolution)

        self.size_area_world = size_area_world
        self.resolution = resolution

        self.lidar = lidar

        self.grid = np.zeros((self.size_area_world[1], self.size_area_world[0], 3), np.uint8)

    def clamp(self, value, min_value, max_value):
        return int(max(min_value, min(value, max_value)))

    def convert_coords(self, coords):
        """
        Convert the coordinates from the world to the grid
        """
        x = self.clamp(coords[0], -self.size_area_world[0] + 1, self.size_area_world[0] - 1)
        y = self.clamp(coords[1], -self.size_area_world[1] + 1, self.size_area_world[1] - 1)
        return int(x - self.size_area_world[0] / 2), int(y - self.size_area_world[1] / 2)

    def update_grid(self, angle, coords):
        """
        lidar : lidar data
        """
        #Reset all 1 to 0
        self.grid = np.zeros((self.size_area_world[1], self.size_area_world[0], 3), np.uint8)

        lidar_dist = self.lidar.get_sensor_values()

        def angle_to_TrueAngle(lidar_angle):
            return 2 * (lidar_angle-90) * np.pi / 180

        # Change every pixel at the end of the lidar ray to 1
        for i in range(len(lidar_dist)):
            if lidar_dist[i] < 250:
                x = int(coords[0] + lidar_dist[i] * np.cos(angle_to_TrueAngle(i) + angle))
                y = int(coords[1] + lidar_dist[i] * np.sin(angle_to_TrueAngle(i) + angle))
                y, x = self.convert_coords((x, y))
                self.grid[(x ,y)] = [255, 255, 255]
        return

    def display(self, position, title="lidar grid"):
        """
        Display the grid
        """
        # Draw the position of the drone
        self.grid[self.convert_coords((int(position[0]), int(position[1])))] = [0, 0, 255]

        #Make sure the grid orientation is correct
        self.grid = np.flip(self.grid, 0)

        # Display the image
        cv2.imshow(title, self.grid)
        cv2.waitKey(1)


class MyDroneMapping(DroneAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.iteration: int = 0

        self.estimated_pose = Pose()

        resolution = 8
        self.grid = LidarVisualization(size_area_world=self.size_area,
                                  resolution=resolution,
                                  lidar=self.lidar())

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def control(self):
        """
        We only send a command to do nothing
        """
        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}

        # increment the iteration counter
        self.iteration += 1

        if self.iteration % 10 == 0:
            self.grid.update_grid(self.measured_compass_angle(), self.measured_gps_position())
            self.grid.display(self.measured_gps_position(), title="lidar grid")

        return command


class MyMapMapping(MapAbstract):

    def __init__(self):
        super().__init__()

        # PARAMETERS MAP
        self._size_area = (1113, 750)

        self._rescue_center = RescueCenter(size=(210, 90))
        self._rescue_center_pos = ((440, 315), 0)

        self._number_drones = 1
        self._drones_pos = [((-50, 0), 0)]
        self._drones = []

    def construct_playground(self, drone_type: Type[DroneAbstract]) -> Playground:
        playground = ClosedPlayground(size=self._size_area)

        # RESCUE CENTER
        playground.add_interaction(CollisionTypes.GEM,
                                   CollisionTypes.ACTIVABLE_BY_GEM,
                                   wounded_rescue_center_collision)

        playground.add(self._rescue_center, self._rescue_center_pos)

        add_walls(playground)
        add_boxes(playground)

        # POSITIONS OF THE DRONES
        misc_data = MiscData(size_area=self._size_area,
                             number_drones=self._number_drones)
        for i in range(self._number_drones):
            drone = drone_type(identifier=i, misc_data=misc_data)
            self._drones.append(drone)
            playground.add(drone, self._drones_pos[i])

        return playground


def main():
    my_map = MyMapMapping()
    playground = my_map.construct_playground(drone_type=MyDroneMapping)

    gui = GuiSR(playground=playground,
                the_map=my_map,
                use_keyboard=True,
                )
    gui.run()


if __name__ == '__main__':
    main()
