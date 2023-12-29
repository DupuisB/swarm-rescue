import os
import sys
import cv2 as cv
from typing import List, Type
import arcade
import math

from spg.utils.definitions import CollisionTypes
from spg.playground import Playground

# This line add, to sys.path, the path to parent path of this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # noqa

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.rescue_center import RescueCenter, wounded_rescue_center_collision
from spg_overlay.entities.wounded_person import WoundedPerson
from spg_overlay.gui_map.closed_playground import ClosedPlayground
from spg_overlay.gui_map.gui_sr import GuiSR
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.path import Path
from spg_overlay.utils.pose import Pose
from spg_overlay.utils.utils import clamp, normalize_angle

from maps.walls_medium_02 import add_walls, add_boxes
from solutions.utils.graph import Node, Graph

import numpy as np
import time
from enum import Enum


def nothing(x):
    pass


EMPTY = {"forward": 0, "rotation": 0}
START = np.array([-80, -80])

class State(Enum):
    EXPLORATION = 0
    RESCUE = 1
    RETURN = 2
    START = 3


class ExplorationState(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class Movement(Enum):
    ROTATING = 0
    MOVING = 1
    STOPPED = 2


class MyTestDrone(DroneAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nb_of_iter_counter = 0
        # delta x / delta y par rapport à la position actuelle
        self.position_consigne = [0.0, 100.0]
        self.iter = 0

        self.constants = {"threshold": 130, "minLineLength": 10,
                          "maxLineGap": 10, "max_degre": 30, "max_delta": 30, "points_size": 5,
                          "second_threshold": 100, "second_minLineLength": 10, "second_maxLineGap": 10}

        # Drawing stuff
        self.lines = []
        self.corners = []
        self.graph = Graph()
        self.target_position = START
        self.prev_diff_distance = 0
        self.prev_diff_angle = 0
        self.origine = None
        self.timeObjective = 0

        self.queue = []
        self.state = State.START

        # Goals
        self.origine = None
        self.target_position = np.array([-80, -80])
        self.old_target_positions = []

        # Better Movement
        self.rotationLock = False
        self.rotationValue = 0

        self.objIsNew = True  # Hacky way to know when to rotate in place

        self.movementState = Movement.STOPPED

        # Exploration stuff
        self.explorationState = ExplorationState.NORTH
        self.node = None

        # Nombre de transitions empruntées
        self.trans = 0

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def world_to_absolute2(self, points):
        """
        Convertit les coordonnées monde en coordonnées absolue
        """
        points[:, 0] += self.size_area[0] / 2
        points[:, 1] += self.size_area[1] / 2
        return points

    def world_to_absolute(self, x, y):
        """
        Convertit les coordonnées monde en coordonnées absolues
        """
        x += self.size_area[0] / 2
        y += self.size_area[1] / 2
        return x, y

    def absolute_to_world(self, x, y):
        """
        Convertit les coordonnées absolues en coordonnées monde
        """
        x -= self.size_area[0] / 2
        y -= self.size_area[1] / 2
        return x, y

    def update_map(self, fusion=False):

        def lidar_to_absolute():
            """
            Convertit les points du lidar en coordonnées monde
            """
            lidar_vals = self.lidar_values()
            mask = lidar_vals < 295
            nb_points = np.sum(mask)
            lidar_points = np.zeros((nb_points, 2))
            lidar_points[:, 0] = lidar_vals[mask] * \
                                 np.cos(self.lidar_rays_angles()[mask] +
                                        self.measured_compass_angle()) + self.measured_gps_position()[0]
            lidar_points[:, 1] = lidar_vals[mask] * \
                                 np.sin(self.lidar_rays_angles()[mask] +
                                        self.measured_compass_angle()) + self.measured_gps_position()[1]
            return lidar_points

        def plot_points(points):
            """
            Affiche les points sur une image, points donnees en coordonnees absolues
            """
            img = np.zeros((self.size_area[1], self.size_area[0], 1), np.uint8)
            for point in points:
                cv.circle(img, (int(point[0]), int(point[1])), self.constants["points_size"], (255, 255, 255), -1)
            return img

        def detect_lines_from_points(points):
            """
            Detection des lignes, points donnees en coordonnees absolues
            """
            points_img = plot_points(points)
            lines = cv.HoughLinesP(points_img, 1, np.pi / 180, self.constants["threshold"],
                                   minLineLength=self.constants["minLineLength"],
                                   maxLineGap=self.constants["maxLineGap"])
            return lines

        def detect_lines_from_img(img):
            """
            Detection des lignes, img donnees en coordonnees absolues
            """
            lines = cv.HoughLinesP(img, 1, np.pi / 180, self.constants["second_threshold"],
                                   minLineLength=self.constants["second_minLineLength"],
                                   maxLineGap=self.constants["second_maxLineGap"])
            return lines

        def align_lines_on_axis(lines):
            """
            Aligne les lignes sur les axes
            """

            if lines is None:
                return None, None

            # on calcule l'angle de chaque ligne, si le dénominateur est nul on met l'angle à pi/2
            # on crée un mask pour éviter les divisions par 0

            # on met l'angle à pi/2 pour les lignes verticales
            @np.vectorize
            def curstom_arctan(a, b): return np.pi / \
                                             2 if b == 0 else abs(np.arctan(a / b))

            angles = curstom_arctan(
                lines[:, 0, 3] - lines[:, 0, 1], lines[:, 0, 2] - lines[:, 0, 0])
            # convertion de l'angle maximum en radians
            eps = np.pi / 180 * self.constants["max_degre"]
            # une ligne est horizontale si angle < eps
            mask_horizontal = angles < eps
            # une ligne est verticale si angle > pi/2 - eps
            mask_vertical = angles > np.pi / 2 - eps

            lignes_horizontales = lines[mask_horizontal]
            lignes_verticales = lines[mask_vertical]
            # pour chaque ligne horizontale on calcule son y moyen
            y_moyens = np.array(
                [lignes_horizontales[:, 0, 1] + lignes_horizontales[:, 0, 3]]) / 2
            x_moyens = np.array(
                [lignes_verticales[:, 0, 0] + lignes_verticales[:, 0, 2]]) / 2
            # pour les lignes horizontales on garde comme coordonnées x celles déjà existantes et on prend comme coordonnées y le y moyen
            lignes_horizontales[:, 0, 1] = y_moyens
            lignes_horizontales[:, 0, 3] = y_moyens
            # pour les lignes verticales on garde comme coordonnées y celles déjà existantes et on prend comme coordonnées x le x moyen
            lignes_verticales[:, 0, 0] = x_moyens
            lignes_verticales[:, 0, 2] = x_moyens
            # on ne garde que les lignes horizontales et verticales
            # on fusionne les lignes horizontales et verticales
            # on retourne les lignes
            return lignes_horizontales, lignes_verticales

        def detect_murs():
            """
            Detection des murs
            """
            lidar_values = np.array(self.lidar_values())
            lidar_angles = np.array(self.lidar_rays_angles())
            mask = lidar_values < 290

            lidar_distance = lidar_values[mask]
            lidar_angles = lidar_angles[mask] + self.measured_compass_angle()

            lidar_x = lidar_distance * np.cos(lidar_angles) + self.measured_gps_position()[0] + self.size_area[0] / 2
            lidar_y = lidar_distance * np.sin(lidar_angles) + self.measured_gps_position()[1] + self.size_area[1] / 2

            img = np.zeros((self.size_area[0], self.size_area[1], 1), np.uint8)
            for i in range(len(lidar_x)):
                cv.circle(img, (int(lidar_x[i]), int(
                    lidar_y[i])), 5, (255, 255, 255), -1)

            lines = cv.HoughLinesP(img, 1, np.pi / 180, 100,
                                   minLineLength=10, maxLineGap=10)
            assert (lines is not None)

            img2 = np.zeros((self.size_area[0], self.size_area[1], 1), np.uint8)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(img2, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # cv.imwrite('lines.png', img2)
            return lines

        def fuse_lines(lines):
            # check si les extrémités des lignes sont proches
            # si oui, on les fusionne

            # on calcule la distance entre les extrémités de chaque ligne
            # si la distance est inférieure à un seuil, on fusionne les lignes
            nombre_de_passes = 10
            max_distance_squared = 625  # 25**2
            for _ in range(nombre_de_passes):
                already_fused = np.zeros(len(lines), dtype=bool)
                new_lines = []
                for i in range(len(lines)):
                    if not already_fused[i]:

                        # on crée un masque pour ne regarder que les lignes non sélectionnées
                        # et qui sont proches de la ligne courante
                        # on prend aussi la ligne courante
                        mask = np.logical_and(
                            np.logical_not(already_fused),
                            np.logical_and(
                                np.abs(lines[:, 0, 0] ** 2 + lines[:, 0, 1] ** 2 -
                                       lines[i, 0, 0] ** 2 - lines[i, 0, 1] ** 2) < max_distance_squared,
                                np.abs(lines[:, 0, 2] ** 2 + lines[:, 0, 3] ** 2 -
                                       lines[i, 0, 2] ** 2 - lines[i, 0, 3] ** 2) < max_distance_squared
                            )
                        )
                        already_fused[mask] = True
                        # on fusionne les lignes sélectionnées
                        if np.sum(mask) >= 1:
                            new_lines.append(np.mean(lines[mask], axis=0))
                lines = np.array(new_lines)

            return lines

        def fuse_close_lines(lines, axis):
            # fusionne les lignes (horizontales ou verticales) qui sont proches selon l'axe donné
            # on considère que deux lignes sont proches si la valeur selon l'axe de leur premier point est proche
            # axis vaut 0 pour les verticales et 1 pour les horizontales (les horizontales ont un y constant)
            if lines is None:
                return None

            already_fused = np.zeros(len(lines), dtype=bool)
            new_lines = []
            for i in range(len(lines)):
                if not already_fused[i]:
                    # on crée un masque pour ne regarder que les lignes non sélectionnées
                    # et qui sont proches de la ligne courante
                    # on prend aussi la ligne courante
                    mask = np.logical_and(
                        np.logical_not(already_fused),
                        np.abs(lines[:, 0, axis] - lines[i, 0, axis]
                               ) < self.constants["max_delta"]
                    )
                    already_fused[mask] = True
                    # on fusionne les lignes sélectionnées
                    if np.sum(mask) >= 1:
                        # prend la moyenne selon l'axe
                        # prend la plus longue selon l'autre axe
                        closes = lines[mask]
                        longs = np.abs(
                            closes[:, 0, (1 - axis)] - closes[:, 0, (1 - axis) + 2])
                        # print(longs.shape, mask.shape)
                        # print(np.argmax(longs, axis=0))
                        extrems = closes[np.argmax(longs)][0]
                        # print(extrems)
                        # extrems = lines[np.argmax(
                        #    abs(lines[mask, 0, (1-axis) + 2]-lines[mask, 0, (1-axis)]))][0]
                        # fait la moyenne des coordonnées des lignes sélectionnées selon l'axe
                        coord = map(lambda x: x[axis], closes[0])
                        med = np.mean(list(coord))
                        # print(med)
                        # print(extrems)
                        new = np.zeros((1, 4))
                        if axis == 0:
                            new[0, 0] = med
                            new[0, 2] = med
                            new[0, 1] = extrems[1]
                            new[0, 3] = extrems[3]

                        else:
                            new[0, 1] = med
                            new[0, 3] = med
                            new[0, 0] = extrems[0]
                            new[0, 2] = extrems[2]

                        new_lines.append(new)

                        # new_lines.append(np.mean(lines[mask], axis=0))
            lines = np.array(new_lines)
            # print(lines.shape)
            if lines.shape[0] == 0:
                return None
            return lines

        def int_coordonates(lines):
            if lines is None:
                return None
            # arrondie les coordonnées des lignes aux entiers
            lines[:, 0, 0] = np.round(lines[:, 0, 0])
            lines[:, 0, 1] = np.round(lines[:, 0, 1])
            lines[:, 0, 2] = np.round(lines[:, 0, 2])
            lines[:, 0, 3] = np.round(lines[:, 0, 3])
            return lines

        def ten_pixels(pos, x1, y1, dist=20):
            if pos[0] <= x1:
                x1 -= dist
            else:
                x1 += dist
            if pos[1] <= y1:
                y1 -= dist
            else:
                y1 += dist
            return x1, y1

        def detect_corners(lines, pos):
            pos = self.world_to_absolute(pos[0], pos[1])
            if lines is None:
                return
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    found_corner = False
                    x1, y1, x2, y2 = lines[i][0]
                    x3, y3, x4, y4 = lines[j][0]
                    # Make sure the lines are basically perpendicular by checking the angle
                    if abs(np.dot(np.array((x2 - x1, y2 - y1)), np.array((x4 - x3, y4 - y3)))) < 100:
                        if abs(x1 - x3) < 5 and abs(y2 - y4) < 5:
                            # Add the corner 10 pixels away from both lines
                            x, y = (x1 + x3) / 2, (y2 + y4) / 2
                            found_corner = True
                        elif abs(x2 - x4) < 5 and abs(y1 - y3) < 5:
                            x, y = (x2 + x4) / 2, (y1 + y3) / 2
                            found_corner = True
                        elif abs(x1 - x4) < 5 and abs(y2 - y3) < 5:
                            x, y = (x1 + x4) / 2, (y2 + y3) / 2
                            found_corner = True
                        elif abs(x2 - x3) < 5 and abs(y1 - y4) < 5:
                            x, y = (x2 + x3) / 2, (y1 + y4) / 2
                            found_corner = True
                        if found_corner:
                            x, y = ten_pixels(pos, x, y)
                            new_node = Node(x, y, primary=True)
                            coords_monde = self.absolute_to_world(x, y)
                            # TODO: update N E S W
                            if self.measured_gps_position()[0] < x:
                                new_node.directions["E"] = 1
                            else:
                                new_node.directions["W"] = 1
                            if self.measured_gps_position()[1] < y:
                                new_node.directions["N"] = 1
                            else:
                                new_node.directions["S"] = 1
                            self.corners.append((x, y))
                            self.graph.add_node(new_node)

        absolute_points = lidar_to_absolute()
        points = self.world_to_absolute2(absolute_points)

        # première passe, détection des lignes
        base_lines = detect_lines_from_points(points)

        if fusion:  # A voir
            lines_h, lines_v = align_lines_on_axis(base_lines)
            if lines_h is not None and lines_v is not None:
                fused_lines = np.concatenate((lines_h, lines_v))
            else:
                return
        else:
            fused_lines = base_lines

        rounded_lines = int_coordonates(fused_lines)
        self.lines = rounded_lines
        detect_corners(rounded_lines, self.measured_gps_position())

        self.graph.update()
        return

    def control_goto(self):
        """
        target_position is self.target_position, given in world coordinates
        Move the drone to a target position using PID, done when the drone is close enough (20 pixels)
        """

        diff_position = self.target_position - np.asarray(self.measured_gps_position())
        distance = np.linalg.norm(diff_position)

        # self.measured_velocity() gives the velocity in both x and y directions
        if distance < 20:
            # TODO: improve PD, rn its shitty as hell
            self.movementState = Movement.STOPPED
            self.old_target_positions.append(self.target_position)
            self.target_position = None
            self.prev_diff_angle = 0
            self.prev_diff_distance = 0
            print("Arrived at target position")
            return {"forward": 0, "rotation": 0}

        desired_angle = math.atan2(diff_position[1], diff_position[0])

        diff_angle = normalize_angle(desired_angle - self.measured_compass_angle())
        deriv_diff_angle = normalize_angle(diff_angle - self.prev_diff_angle)

        if self.movementState == Movement.STOPPED or self.movementState == Movement.ROTATING:
            return self.control_orientation(str(desired_angle))

        # PD controller for rotation
        Ku_angle = 11.16
        Tu_angle = 2.0
        Kp_angle = 0.8 * Ku_angle
        Kd_angle = Ku_angle * Tu_angle / 40.0
        rotation = (Kp_angle * diff_angle + Kd_angle * deriv_diff_angle)
        rotation = clamp(rotation, -1.0, 1.0)

        deriv_diff_distance = distance - self.prev_diff_distance

        # PD controller for forward movement
        Ku_position = 25 / 100
        Tu_position = 26
        Kp_position = 0.8 * Ku_position
        Kd_position = Ku_position * Tu_position / 10.0
        forward = Kp_position * distance + Kd_position * deriv_diff_distance

        forward = clamp(forward, -1.0, 1.0)

        # print(f"forward = {Kp_position * diff_position[0]} + {Kd_position * deriv_diff_position[0]}")

        self.prev_diff_angle = diff_angle
        self.timeObjective += 1

        return {"forward": forward, "rotation": rotation}

    def control_orientation(self, direction):
        """
        Rotate on itself to reach the given direction (using PD)
        """

        dir_to_angle = {"N": np.pi/2, "E": 0, "S": 3*np.pi/2, "W": np.pi}

        if direction in dir_to_angle:
            angle = dir_to_angle[direction]
        else:
            angle = eval(direction)

        if self.movementState == Movement.STOPPED:
            self.movementState = Movement.ROTATING
            self.rotationValue = direction

        diff_angle = normalize_angle(angle - self.measured_compass_angle())

        if abs(diff_angle) < 0.05:
            self.movementState = Movement.MOVING
            print("Now facing", direction)
            return {"forward": 0, "rotation": 0}

        deriv_diff_angle = normalize_angle(diff_angle - self.prev_diff_angle)

        # PID controller for rotation
        Ku_angle = 11.16
        Tu_angle = 2.0
        Kp_angle = 0.8 * Ku_angle
        Kd_angle = Ku_angle * Tu_angle / 40.0
        rotation = Kp_angle * diff_angle + Kd_angle * deriv_diff_angle
        rotation = clamp(rotation, -1.0, 1.0)

        self.prev_diff_angle = diff_angle
        self.timeObjective += 1

        return {"forward": 0, "rotation": rotation}

    def control_exploreStraight(self, direction):
        """
        Go straight facing angle until a wall is detected.
        Uses PID to control the rotation.
        Instruction is either "N", "E", "S" or "W"
        """
        dir_to_vector = {"N": np.array([0, 1]), "E": np.array([1, 0]), "S": np.array([0, -1]), "W": np.array([-1, 0])}

        if self.movementState == Movement.STOPPED:
            direction_vect = dir_to_vector[direction]
            objective = self.measured_gps_position() + direction_vect * 3000
            objective[0] = clamp(objective[0], -self.size_area[0]/2, self.size_area[0]/2)
            objective[1] = clamp(objective[1], -self.size_area[1]/2, self.size_area[1]/2)
            self.target_position = objective
            return self.control_orientation(direction)

        if self.movementState == Movement.ROTATING:
            return self.control_orientation(direction)

        if self.lidar_values()[90] < 100:
            print('Wall in front !')
            self.movementState = Movement.STOPPED
            return EMPTY

        return self.control_goto()

    def control(self):
        self.update_map()
        self.iter += 1

        angle = 0

        diff_angle = normalize_angle(angle - self.measured_compass_angle())


        if self.state == State.START:
            command = self.control_goto()
            if self.movementState == Movement.STOPPED:
                self.state = State.EXPLORATION
            return command

        ###### EXPLORATION ######
        if self.state == State.EXPLORATION:
            if self.movementState == Movement.STOPPED:
                self.trans += 1
            if self.trans == 1:
                command = self.control_exploreStraight("N")
            elif self.trans == 3:
                command = self.control_exploreStraight("S")
            elif self.trans == 5:
                command = self.control_exploreStraight("W")
            elif self.trans == 7:
                command = self.control_exploreStraight("E")
            elif self.trans % 2 == 0 and self.trans <= 8:
                self.target_position = START
                command = self.control_goto()
            else:
                command = EMPTY
            return command

    def draw_bottom_layer(self):
        # Draw the lines on the arcade playground
        self.graph.draw_arcade()
        if self.lines is not None:
            for line in self.lines:
                x1, y1, x2, y2 = line[0]
                arcade.draw_line(x1, y1, x2, y2, color=arcade.color.RED, line_width=2)
        # Draw a line to the target position
        if self.target_position is not None:
            x1, y1 = self.world_to_absolute(self.measured_gps_position()[0], self.measured_gps_position()[1])
            x2, y2 = self.world_to_absolute(self.target_position[0], self.target_position[1])
            arcade.draw_line(x1, y1, x2, y2,
                             arcade.color.RED, 2)
            arcade.draw_circle_filled(x2, y2, 5, arcade.color.RED)


#####MAPS#####
class MyMapMapping(MapAbstract):

    def __init__(self):
        super().__init__()

        # PARAMETERS MAP
        self._size_area = (1112, 750)

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
    playground = my_map.construct_playground(drone_type=MyTestDrone)

    gui = GuiSR(playground=playground,
                the_map=my_map,
                use_keyboard=False,
                draw_gps=True
                )
    gui.run()


if __name__ == '__main__':
    main()
