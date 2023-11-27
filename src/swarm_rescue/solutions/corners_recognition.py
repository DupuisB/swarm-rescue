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

import arcade

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
from spg_overlay.utils.path import Path


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
        self.path_done = Path()


        self.grid = np.zeros((self.size_area_world[1], self.size_area_world[0], 3), np.uint8)

    def clamp(self, value, min_value, max_value):
        return int(max(min_value, min(value, max_value)))

    def draw_line(self, x1, y1, x2, y2):
        """
        Draw a line on the playground
        """
        # Draw the line
        arcade.draw_line(x1, y1, x2, y2, arcade.color.GREEN, 2)


    def convert_coords(self, coords):
        """
        Convert the coordinates from the world to the grid
        """
        x = self.clamp(coords[0], -self.size_area_world[0]/2 + 1, self.size_area_world[0]/2 - 1)
        y = self.clamp(coords[1], -self.size_area_world[1]/2 + 1, self.size_area_world[1]/2 - 1)
        return int(x + self.size_area_world[0] / 2), int(y + self.size_area_world[1] / 2)

    def convert_coords_inv(self, coords):
        """
        Convert the coordinates from the grid to the world
        """
        x = self.clamp(coords[0], 0, self.size_area_world[0] - 1)
        y = self.clamp(coords[1], 0, self.size_area_world[1] - 1)
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
                self.grid[(x, y)] = [255, 255, 255]
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

    def detectCorners(self):
        """
        Detects the corners of the room
        """
        # Convert the image to grayscale
        gray = cv2.cvtColor(self.grid, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        #morphological operations to connect detected points
        kernel = np.ones((25, 25), np.uint8)
        self.grid = cv2.morphologyEx(self.grid, cv2.MORPH_CLOSE, kernel)

        # Detect the interest points
        interestPoints = cv2.goodFeaturesToTrack(gray, 100, 0.5, 5)
        interestPoints = interestPoints.astype(int)

        for i in interestPoints:
            x, y = i.ravel()
            cv2.circle(self.grid, (x, y), 3, 255, -1)

        # Hough transform to detect lines using interest points
        grid = cv2.cvtColor(self.grid, cv2.COLOR_BGR2GRAY)
        lines = cv2.HoughLinesP(grid, 1, np.pi/180, 100, minLineLength=50, maxLineGap=5)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(self.grid, (x1, y1), (x2, y2), (0, 255, 0), 2)
                self.draw_line(x1, y1, x2, y2)

            # Draw a red circle at lines intersections
            for i in range(len(lines)):
                for j in range(i+1, len(lines)):
                    line1 = lines[i][0]
                    line2 = lines[j][0]
                    p1 = np.array((line1[0], line1[1]))
                    p2 = np.array((line1[2], line1[3]))
                    p3 = np.array((line2[0], line2[1]))
                    p4 = np.array((line2[2], line2[3]))
                    #Make sure the lines are basically perpendicular
                    if np.abs(np.dot(p2-p1, p4-p3)) < 100:
                        # Draw a red circle at the intersection
                        if np.linalg.norm(p1-p3) < 10:
                            cv2.circle(self.grid, (int((p1[0]+p3[0])/2), int((p1[1]+p3[1])/2)), 10, (0, 0, 255), -1)
                        if np.linalg.norm(p1-p4) < 10:
                            cv2.circle(self.grid, (int((p1[0]+p4[0])/2), int((p1[1]+p4[1])/2)), 10, (0, 0, 255), -1)
                        if np.linalg.norm(p2-p3) < 10:
                            cv2.circle(self.grid, (int((p2[0]+p3[0])/2), int((p2[1]+p3[1])/2)), 10, (0, 0, 255), -1)
                        if np.linalg.norm(p2-p4) < 10:
                            cv2.circle(self.grid, (int((p2[0]+p4[0])/2), int((p2[1]+p4[1])/2)), 10, (0, 0, 255), -1)



        # Display the image
        # Flip the image to make sure the orientation is correct
        self.grid = np.flip(self.grid, 0)
        cv2.imshow("Corners", self.grid)
        cv2.waitKey(1)
        return lines



class MyDroneCorners(DroneAbstract):
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
        lines = None
        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}

        # increment the iteration counter
        self.iteration += 1

        if self.iteration % 10 == 0:
            self.grid.update_grid(self.measured_compass_angle(), self.measured_gps_position())
            #self.grid.display(self.measured_gps_position(), title="lidar grid")
            lines = self.grid.detectCorners()

        arcade.draw_point(0, 0, arcade.color.RED, 20)

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
    playground = my_map.construct_playground(drone_type=MyDroneCorners)

    gui = GuiSR(playground=playground,
                the_map=my_map,
                use_keyboard=True,
                use_mouse_measure=False,
                enable_visu_noises=False,
                )

    gui.run()


if __name__ == '__main__':
    main()
