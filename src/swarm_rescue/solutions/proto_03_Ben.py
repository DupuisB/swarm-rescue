import os
import sys
import cv2 as cv
from typing import List, Type
import arcade
import math
import random
from itertools import chain

from typing import Union
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
    EXPLORING_NODE = 1
    RESCUE = 2
    RETURN = 3
    START = 4
    WAITING = 5


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

        self.node_objective: Node = None

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

        #Graph stuff
        self.graph = Graph()
        self.lastNode:Node = None #Either primary or secondary
        self.lastSecondaryNode:Node = None

        #Divers
        self.inRotation = False #Used to know the tick right after a rotation
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

    def world_to_absolute(self, x:Union[int, float, tuple, np.array], y = None):
        """
        Convertit les coordonnées monde en coordonnées absolues
        """
        if y:
            x += self.size_area[0] / 2
            y += self.size_area[1] / 2
            return x, y
        else:
            x[0] += self.size_area[0] / 2
            x[1] += self.size_area[1] / 2
            return x

    def absolute_to_world(self, x:Union[int, float, tuple, np.array], y = None):
        """
        Convertit les coordonnées absolues en coordonnées monde
        """
        if y:
            x -= self.size_area[0] / 2
            y -= self.size_area[1] / 2
            return x, y
        else:
            x[0] -= self.size_area[0] / 2
            x[1] -= self.size_area[1] / 2
            return x

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

        def ten_pixels(pos, x1, y1, dist=30):
            if pos[0] <= x1:
                x1 -= dist
            else:
                x1 += dist
            if pos[1] <= y1:
                y1 -= dist
            else:
                y1 += dist
            return x1, y1

        def visible_nodes():
            """
            return the nodes that are visible from the current position
            returns a list of dictionnaries with the node, the angle and the distance
            """
            visibleNodes = []
            #Scan if some nodes can be connected
            for i in range(len(self.graph.nodes)):
                node = self.graph.nodes[i]
                if node.isPrimary:
                    x, y = self.absolute_to_world(node.x, node.y)
                    #Find the lidar ray for each node
                    angle = np.arctan2(y - self.measured_gps_position()[1], x - self.measured_gps_position()[0])
                    #angles are relative to the drone, so we add the compass angle (0 is East)
                    angle -= self.measured_compass_angle()
                    #lidar is 181 rays, 0 and 180 are the same, 90 is the front, return modulo 180
                    angle = int(np.round(np.rad2deg(angle)/2) + 90) % 180
                    #Calculate the distance between the drone and the nodes
                    dist = np.linalg.norm(np.array((x, y)) - np.array(self.measured_gps_position()))
                    if self.lidar_values()[angle] < 280 and self.lidar_values()[angle] > dist and node.isPrimary:
                        data = (node, angle, dist)
                        visibleNodes.append(data)
                        node.isVisible = True
            return visibleNodes

        def update_transitions():
            """
            Check if transitions are possible between visible nodes
            """
            visibleNodes = visible_nodes()
            for i in range(len(visibleNodes)):
                for j in range(i + 1, len(visibleNodes)):
                    node1, angle1, dist1 = visibleNodes[i]
                    node2, angle2, dist2 = visibleNodes[j]
                    if node1.id == node2.id or node1.x == node2.x or node1 in node2.neighbors:
                        continue
                    #Calculate the equation of the line between the two nodes
                    x1, y1 = self.absolute_to_world(node1.x, node1.y)
                    x2, y2 = self.absolute_to_world(node2.x, node2.y)
                    a = (y2 - y1) / (x2 - x1)
                    b = y1 - a * x1 #Equation is y = ax + b
                    mini, maxi = min(angle1, angle2), max(angle1, angle2)
                    if abs(maxi-mini) <= 90:
                        arg = range(mini, maxi)
                    else:
                        arg = chain(range(maxi, 180), range(0, mini))

                    #Check if all the points between the two nodes are free
                    #We check  the intersection of the line with all the rays between the two nodes
                    for angle in arg:
                        #Calculate the equation of the angle-th ray
                        a2 = np.tan(np.deg2rad(angle))
                        b2 = self.measured_gps_position()[1] - a2 * self.measured_gps_position()[0]
                        #Calculate the intersection of the two lines
                        x = (b2 - b) / (a - a2)
                        #Calculate the distance between x and the drone
                        dist = np.linalg.norm(np.array((x, a2 * x + b2)) - np.array(self.measured_gps_position()))
                        #Threshold is less strict near the actual nodes
                        if abs(angle - min(angle1, angle2)) < 5 and abs(angle - max(angle1, angle2)) < 5:
                            if self.lidar_values()[angle] < dist + 10:
                                #Exit the loop if the ray is blocked
                                break
                            if self.lidar_values()[angle] < 10: #Chiant a expliquer mais important
                                if self.lidar_values()[angle + 90 % 180] < 10 - dist:
                                    break

                        else:
                            if self.lidar_values()[angle] < dist + 30:
                                break
                            if self.lidar_values()[angle] < 20: #Chiant a expliquer mais important
                                if self.lidar_values()[angle + 90 % 180] < 30 - dist:
                                    break
                    else:
                        #If the loop is not broken, the transition is possible
                        node1.neighbors.append(node2)
                        node2.neighbors.append(node1)
                        break

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
                            self.corners.append((x, y))
                            self.graph.add_node(new_node, self.world_to_absolute(self.measured_gps_position()), self.lastSecondaryNode)
                            if self.lastSecondaryNode is not None:
                                self.lastSecondaryNode.neighbors.append(new_node)
                                new_node.neighbors.append(self.lastSecondaryNode)

        def distance_to_segment(x, y, x1, y1, x2, y2):
            """
            Distance between a point and a segment
            """
            # https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
            A = x - x1
            B = y - y1
            C = x2 - x1
            D = y2 - y1

            dot = A * C + B * D
            len_sq = C * C + D * D
            param = -1
            if len_sq != 0:
                param = dot / len_sq

            if param < 0:
                xx = x1
                yy = y1
            elif param > 1:
                xx = x2
                yy = y2
            else:
                xx = x1 + param * C
                yy = y1 + param * D

            dx = x - xx
            dy = y - yy
            return np.sqrt(dx * dx + dy * dy)

        def closest_lines(lines, pos, mini):
            """
            Returns FALSE is at least mini/2 pixels away from the closest line
            """
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dist = distance_to_segment(pos[0], pos[1], x1, y1, x2, y2)
                if dist < mini/2:
                    return False
            return True
        def are_aligned(lines, x1, y1, x2, y2, x3, y3, x4, y4, threshold = 0.01, dist_decalage = 30, minimum_gap = 20):
            v1 = np.array([(x2 - x1), (y2 - y1)]) / np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            v2 = np.array([(x4 - x3), (y4 - y3)]) / np.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2)

            if abs(np.dot(v1, v2) - 1) > threshold: # Checks if the lines are parallel
                return None

            #Project x3, y3 on the line (x1, y1) (x2, y2)
            dist = abs((x2 - x1)*(y1 - y3) - (x1 - x3)*(y2 - y1)) / np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if dist > 5:
                return None

            #Find the closest points in the segments, try every comparison
            p1, p2, mini_dist = None, None, float('inf')
            if np.linalg.norm((x1 - x3, y1 - y3)) < mini_dist:
                p1, p2, mini_dist = (x1, y1), (x3, y3), np.linalg.norm((x1 - x3, y1 - y3))
            if np.linalg.norm((x1 - x4, y1 - y4)) < mini_dist:
                p1, p2, mini_dist = (x1, y1), (x4, y4), np.linalg.norm((x1 - x4, y1 - y4))
            if np.linalg.norm((x2 - x3, y2 - y3)) < mini_dist:
                p1, p2, mini_dist = (x2, y2), (x3, y3), np.linalg.norm((x2 - x3, y2 - y3))
            if np.linalg.norm((x2 - x4, y2 - y4)) < mini_dist:
                p1, p2, mini_dist = (x2, y2), (x4, y4), np.linalg.norm((x2 - x4, y2 - y4))

            if mini_dist < minimum_gap:
                return None

            middle_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

            if closest_lines(lines, middle_point, mini_dist):
                perpendicular_vector = np.array([-v1[1], v1[0]])  # Perpendicular vector
                # We want the perpendicular vector to be oriented towards the drone half plane
                x, y = self.measured_gps_position()
                #Convert middle point to world coordinates
                middle_point2 = self.absolute_to_world(middle_point[0], middle_point[1])
                if np.dot(perpendicular_vector, np.array([x - middle_point2[0], y - middle_point2[1]])) < 0:
                    perpendicular_vector *= -1
                middle_point += dist_decalage * perpendicular_vector
                return middle_point
            return None

        def detect_gaps(lines, pos):
            pos = self.world_to_absolute(pos[0], pos[1])
            if lines is None:
                return
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    found_corner = False
                    x1, y1, x2, y2 = lines[i][0]
                    x3, y3, x4, y4 = lines[j][0]
                    # Check if the lines are basically aligned
                    data = are_aligned(lines, x1, y1, x2, y2, x3, y3, x4, y4)
                    if data is not None:
                        new_node = Node(data[0], data[1], primary=True)
                        new_node.isGap = True
                        coords_monde = self.absolute_to_world(data[0], data[1])
                        self.graph.add_node(new_node, self.world_to_absolute(self.measured_gps_position()),
                                            self.lastSecondaryNode)
                        if self.lastSecondaryNode is not None:
                            self.lastSecondaryNode.neighbors.append(new_node)
                            new_node.neighbors.append(self.lastSecondaryNode)

        def update_gaps(lines):
            for node in self.graph.nodes:
                if node.isGap:
                    if not closest_lines(lines, (node.x, node.y), 70):
                        if node.weight <= 3:
                            self.graph.nodes.remove(node)
                        else:
                            node.weight -= 3

        absolute_points = lidar_to_absolute()
        points = self.world_to_absolute2(absolute_points)

        # première passe, détection des lignes
        base_lines = detect_lines_from_points(points)

        if fusion:  # A voir
            lines_h, lines_v = align_lines_on_axis(base_lines)
            lines_h, lines_v = fuse_close_lines(lines_h, 1), fuse_close_lines(lines_v, 0)
            if lines_h is not None and lines_v is not None:
                fused_lines = np.concatenate((lines_h, lines_v))
            else:
                return
        else:
            fused_lines = base_lines

        rounded_lines = int_coordonates(fused_lines)
        self.lines = rounded_lines
        detect_corners(rounded_lines, self.measured_gps_position())
        detect_gaps(rounded_lines, self.measured_gps_position())

        #Every 15 frames, we add the current position as a secondary node
        if self.iter % 15 == 0:
            new_node = Node(self.world_to_absolute(self.measured_gps_position()), primary=False)
            self.graph.add_node(new_node)
            if self.lastSecondaryNode is not None:
                self.lastSecondaryNode.neighbors.append(new_node)
                new_node.neighbors.append(self.lastSecondaryNode)
            self.lastSecondaryNode = new_node

        update_transitions()
        update_gaps(rounded_lines)
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

        if not self.inRotation:
            self.inRotation = True

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
            if self.inRotation:
                #Update the direction of the node
                self.node_objective.directions[direction] = -1
                self.inRotation = False
            print('Wall in front !')
            #Add the node to the graph
            new_node = Node(self.world_to_absolute(self.measured_gps_position()), primary=False)
            self.graph.add_node(new_node)
            new_node.neighbors.append(self.node_objective)
            self.node_objective.neighbors.append(new_node)
            self.movementState = Movement.STOPPED
            return EMPTY

        if self.inRotation:
            #Update the direction of the node
            self.node_objective.directions[direction] = 2
            self.inRotation = False

        return self.control_goto()

    def control_gotoNode(self):
        """
        Go to the given node
        """
        self.target_position = self.absolute_to_world(self.node_objective.x, self.node_objective.y)
        command = self.control_goto()
        if self.movementState == Movement.STOPPED:
            self.node_objective.neighbors.append(self.lastSecondaryNode)
            self.lastSecondaryNode.neighbors.append(self.node_objective)
        self.lastNode = self.node_objective
        if not self.node_objective.isPrimary:
            self.lastSecondaryNode = self.node_objective
        return command

    def control_explore_node(self):
        """
        Explore the given node
        """
        if self.movementState == Movement.STOPPED:
            self.trans += 1
        if self.trans == 1:
            self.state = State.EXPLORING_NODE
            command = self.control_exploreStraight("N")
        elif self.trans == 3:
            command = self.control_exploreStraight("S")
        elif self.trans == 5:
            command = self.control_exploreStraight("W")
        elif self.trans == 7:
            command = self.control_exploreStraight("E")
        elif self.trans % 2 == 0 and self.trans <= 8:
            command = self.control_gotoNode()
        else:
            self.movementState = Movement.STOPPED
            self.state = State.START
            command = EMPTY
        return command

    def control(self):
        self.update_map()
        self.iter += 1

        if self.state == State.START:
            #print('Choosing a node to explore')
            #Pick a random unvisited node

            i = 0
            while self.node_objective is None or not self.node_objective.isPrimary or self.graph.visited[self.node_objective.id]:
                if i == len(self.graph.nodes):
                    break
                self.node_objective = self.graph.nodes[i]
                i += 1
            else:
                command = self.control_gotoNode()
                if self.movementState == Movement.STOPPED:
                    self.state = State.WAITING
                return command
            return EMPTY

        ###### EXPLORATION ######
        if self.state == State.WAITING: #WAITING AT NODE
            print("Starting exploration")
            self.trans = 0
            self.graph.visited[self.node_objective.id] = True
            command = self.control_explore_node()
            return command

        if self.state == State.EXPLORING_NODE:
            command = self.control_explore_node()
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
                use_keyboard=True,
                draw_gps=True
                )
    gui.run()


if __name__ == '__main__':
    main()
