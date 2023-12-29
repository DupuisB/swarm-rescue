import os
import sys
import cv2 as cv
from typing import List, Type
import arcade

from spg.utils.definitions import CollisionTypes
from spg.playground import Playground

# This line add, to sys.path, the path to parent path of this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # noqa

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.rescue_center import RescueCenter, wounded_rescue_center_collision
from spg_overlay.entities.wounded_person import WoundedPerson
from spg_overlay.gui_map.closed_playground import ClosedPlayground
from spg_overlay.gui_map.gui_sr import GuiSR
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.path import Path
from spg_overlay.utils.pose import Pose
from spg_overlay.utils.utils import clamp

from maps.walls_medium_02 import add_walls, add_boxes
from solutions.utils.graph import Node, Graph

import numpy as np
import time


def nothing(x):
    pass


class MyTestDrone(DroneAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nb_of_iter_counter = 0
        # delta x / delta y par rapport à la position actuelle
        self.position_consigne = [0.0, 100.0]
        self.iter = 0
        self.prev_diff_position = 0

        self.constants = {"threshold": 130, "minLineLength": 10,
                          "maxLineGap": 10, "max_degre": 30, "max_delta": 30, "points_size": 5,
                          "second_threshold": 100, "second_minLineLength": 10, "second_maxLineGap": 10}

        # Drawing stuff
        self.lines = []
        self.corners = []
        self.graph = Graph()

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def lidar_to_absolute(self):
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

    def world_to_absolute2(self, points):
        """
        Convertit les coordonnées monde en coordonnées absolue
        """
        points[:, 0] += self.size_area[0]/2
        points[:, 1] += self.size_area[1]/2
        return points

    def world_to_absolute(self, x, y):
        """
        Convertit les coordonnées monde en coordonnées absolues
        """
        x += self.size_area[0]/2
        y += self.size_area[1]/2
        return x, y

    def absolute_to_world(self, x, y):
        """
        Convertit les coordonnées absolues en coordonnées monde
        """
        x -= self.size_area[0]/2
        y -= self.size_area[1]/2
        return x, y

    def plot_points(self, points):
        """
        Affiche les points sur une image, points donnees en coordonnees absolues
        """
        img = np.zeros((self.size_area[1], self.size_area[0], 1), np.uint8)
        for point in points:
            cv.circle(img, (int(point[0]), int(point[1])), self.constants["points_size"], (255, 255, 255), -1)
        return img

    def detect_lines_from_points(self, points):
        """
        Detection des lignes, points donnees en coordonnees absolues
        """
        points_img = self.plot_points(points)
        lines = cv.HoughLinesP(points_img, 1, np.pi / 180, self.constants["threshold"],
                               minLineLength=self.constants["minLineLength"], maxLineGap=self.constants["maxLineGap"])
        return lines

    def detect_lines_from_img(self, img):
        """
        Detection des lignes, img donnees en coordonnees absolues
        """
        lines = cv.HoughLinesP(img, 1, np.pi / 180, self.constants["second_threshold"],
                               minLineLength=self.constants["second_minLineLength"],
                               maxLineGap=self.constants["second_maxLineGap"])
        return lines

    def align_lines_on_axis(self, lines):
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

    def detect_murs(self):
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

        #cv.imwrite('lines.png', img2)
        return lines

    def fuse_lines(self, lines):
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

    def fuse_close_lines(self, lines, axis):
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

    def int_coordonates(self, lines):
        if lines is None:
            return None
        # arrondie les coordonnées des lignes aux entiers
        lines[:, 0, 0] = np.round(lines[:, 0, 0])
        lines[:, 0, 1] = np.round(lines[:, 0, 1])
        lines[:, 0, 2] = np.round(lines[:, 0, 2])
        lines[:, 0, 3] = np.round(lines[:, 0, 3])
        return lines

    def ten_pixels(self, pos, x1, y1, dist=20):
        if pos[0] <= x1:
            x1 -= dist
        else:
            x1 += dist
        if pos[1] <= y1:
            y1 -= dist
        else:
            y1 += dist
        return x1, y1

    def detect_corners(self, lines, pos):
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
                        x, y = self.ten_pixels(pos, x, y)
                        new_node = Node(x, y, primary=True)
                        print("New corner: ", x, y)
                        coords_monde = self.absolute_to_world(x, y)
                        print("Monde: ", coords_monde)
                        #TODO: update N E S W
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

    def draw_lines(self, lines):
        img = np.zeros(
            (self.size_area[0], self.size_area[1], 1), np.uint8)
        if lines is None:
            return img
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(img, (int(x1), int(y1)), (int(x2),
                                              int(y2)), (255, 255, 255), 2)
        return img

    def control(self):
        command = {"forward": 0.0,
                   "lateral": 0.0, }

        absolute_points = self.lidar_to_absolute()
        points = self.world_to_absolute2(absolute_points)

        # première passe, détection des lignes
        base_lines = self.detect_lines_from_points(points)

        fusion = False
        if fusion:
            lines_h, lines_v = self.align_lines_on_axis(base_lines)
            if lines_h is not None and lines_v is not None:
                fused_lines = np.concatenate((lines_h, lines_v))
            else:
                return command
        else:
            fused_lines = base_lines

        rounded_lines = self.int_coordonates(fused_lines)
        self.lines = rounded_lines
        self.detect_corners(rounded_lines, self.measured_gps_position())

        self.iter += 1
        self.graph.update()
        return command

    def draw_bottom_layer(self):
        # Draw the lines on the arcade playground
        self.graph.draw_arcade()
        if self.lines is None:
            return
        for line in self.lines:
            x1, y1, x2, y2 = line[0]
            arcade.draw_line(x1, y1, x2, y2, color=arcade.color.RED, line_width=2)
        #for corner in self.corners:
            #x, y = corner
            #arcade.draw_circle_filled(x, y, 15, arcade.color.BLUE)


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
                )
    gui.run()


if __name__ == '__main__':
    main()
