import os
import sys
import cv2 as cv
from typing import List, Type

from spg.utils.definitions import CollisionTypes

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
        self.window_debug = cv.namedWindow('steps')
        self.window_constants = cv.namedWindow('constants')

        self.constants = {"threshold": 100, "minLineLength": 10,
                          "maxLineGap": 10, "max_degre": 30, "max_delta": 30, "points_size": 5,
                          "second_threshold": 100, "second_minLineLength": 10, "second_maxLineGap": 10}

        for c in self.constants:
            cv.createTrackbar(
                c, 'constants', self.constants[c], 10*self.constants[c], nothing)

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def convert_lidar_points_to_absolute(self):
        """
        Convertit les points du lidar en coordonnées absolues
        """
        lidar_vals = self.lidar_values()
        # we create a mask to keep only the points that are not too far

        mask = lidar_vals < 290
        # calculate the number of points that are not too far
        nb_points = np.sum(mask)
        # we create a matrix of size nb_points x 2
        # the first column will contain the x coordinates
        # the second column will contain the y coordinates
        lidar_points = np.zeros((nb_points, 2))
        # we fill the matrix
        lidar_points[:, 0] = lidar_vals[mask] * \
            np.cos(self.lidar_rays_angles()[mask] +
                   self.measured_compass_angle())
        lidar_points[:, 1] = lidar_vals[mask] * \
            np.sin(self.lidar_rays_angles()[mask] +
                   self.measured_compass_angle())
        # we add the position of the drone
        lidar_points[:, 0] += self.measured_gps_position()[0]

        lidar_points[:, 1] += self.measured_gps_position()[1]

        return lidar_points

    def convert_absolute_points_to_image_points(self, points):
        # la première colonne recoit +self.size_area[0]/2+25
        # la deuxième colonne recoit +self.size_area[1]/2+25
        points[:, 0] += self.size_area[0]/2+25
        points[:, 1] *= -1
        # sur une image le 0,0 est en haut à gauche
        points[:, 1] += self.size_area[1] / 2+25
        return points

    def plot_points(self, points):
        """
        Affiche les points sur une image
        """
        # we create a black image
        img = np.zeros(
            (self.size_area[0]+50, self.size_area[1]+50, 1), np.uint8)
        # we add white circles at the coordinates of the points
        for i in range(points.shape[0]):
            cv.circle(img, (int(points[i, 0]), int(
                points[i, 1])), self.constants["points_size"], (255, 255, 255), -1)
        return img

    def detect_lines_from_points(self, points):
        points_img = self.plot_points(points)
        # we detect the lines
        lines = cv.HoughLinesP(points_img, 1, np.pi/180, self.constants["threshold"],
                               minLineLength=self.constants["minLineLength"], maxLineGap=self.constants["maxLineGap"])
        return lines

    def detect_lines_from_img(self, img):
        # we detect the lines
        lines = cv.HoughLinesP(img, 1, np.pi/180, self.constants["second_threshold"],
                               minLineLength=self.constants["second_minLineLength"], maxLineGap=self.constants["second_maxLineGap"])
        return lines

    def align_lines_on_axis(self, lines):
        # aligne les lignes en considérant qu'elles sont soit horizontales, soit verticales
        # on considère que les lignes sont horizontales si leur angle est proche de 0 ou de pi
        # on considère que les lignes sont verticales si leur angle est proche de pi/2 ou de 3*pi/2

        if lines is None:
            return None, None

        # angle = abs(arctan(dy/dx))
        # on calcule l'angle de chaque ligne, si le dénominateur est nul on met l'angle à pi/2
        # on crée un mask pour éviter les divisions par 0

        # on met l'angle à pi/2 pour les lignes verticales
        @np.vectorize
        def curstom_arctan(a, b): return np.pi/2 if b == 0 else np.arctan(a/b)
        angles = curstom_arctan(
            lines[:, 0, 3]-lines[:, 0, 1], lines[:, 0, 2]-lines[:, 0, 0])
        # convertion de l'angle maximum en radians
        eps = np.pi/180*self.constants["max_degre"]
        # une ligne est horizontale si angle < eps
        mask_horizontal = angles < eps
        # une ligne est verticale si angle > pi/2 - eps
        mask_vertical = angles > np.pi/2 - eps

        lignes_horizontales = lines[mask_horizontal]
        lignes_verticales = lines[mask_vertical]
        # pour chaque ligne horizontale on calcule son y moyen
        y_moyens = np.array(
            [lignes_horizontales[:, 0, 1] + lignes_horizontales[:, 0, 3]])/2
        x_moyens = np.array(
            [lignes_verticales[:, 0, 0] + lignes_verticales[:, 0, 2]])/2
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
        # transforme les informations du lidar en coordonnées cartésiennes
        # les données du lidar sont relatives au drone, on a les distances et les angles relatives au drone
        # on veut les coordonnées cartésiennes par rapport au monde
        # on utilise la position gps du drone pour ça
        taille_carte = self.size_area
        drone_angle = self.measured_compass_angle()
        lidar_data = np.array([(d, a) for d, a in zip(
            self.lidar_values(), self.lidar_rays_angles()) if d < 290])
        # récupère les tableaux des distances et des angles correspondant
        lidar_distance = np.array(list(map(lambda x: x[0], lidar_data)))
        lidar_angles = np.array(list(map(lambda x: x[1], lidar_data)))
        # print(lidar_distance)
        # print(lidar_angles)
        lidar_x = []
        lidar_y = []
        for i in range(len(lidar_distance)-1):
            lidar_x.append(
                lidar_distance[i] * np.cos(lidar_angles[i]+self.measured_compass_angle()) + self.measured_gps_position()[0] + taille_carte[0]/2+25)
            lidar_y.append(
                lidar_distance[i] * np.sin(lidar_angles[i]+self.measured_compass_angle()) + self.measured_gps_position()[1] + taille_carte[1]/2+25)
        # on a maintenant les coordonnées cartésiennes des points détectés par le lidar
        # on transforme ça en image binaire

        img = np.zeros((taille_carte[0]+50, taille_carte[1]+50, 1), np.uint8)
        # on ajoute des cercles blancs aux coordonnées des points détectés par le lidar
        for i in range(len(lidar_x)):
            # on met un pixel blanc
            # img[int(lidar_x[i]), int(lidar_y[i])] = 255
            # on met un cercle blanc
            cv.circle(img, (int(lidar_x[i]), int(
                lidar_y[i])), 5, (255, 255, 255), -1)

        # on affiche l'image
        # cv.imshow("img", img)
        # cv.imwrite('points.png', img)

        # on detecte les lignes
        # on utilise la transformée de Hough
        # edges = cv.Canny(img, 50, 200, apertureSize=3)
        # cv.imwrite('edges.png', edges)
        lines = cv.HoughLinesP(img, 1, np.pi/180, 100,
                               minLineLength=10, maxLineGap=10)
        # print(lines)
        assert (lines is not None)
        # on trace ces lignes dans une noubelle image
        img2 = np.zeros((taille_carte[0]+50, taille_carte[1]+50, 1), np.uint8)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(img2, (x1, y1), (x2, y2), (255, 255, 255), 2)
        # on enregistre l'image
        cv.imwrite('lines.png', img2)
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
                            np.abs(lines[:, 0, 0]**2 + lines[:, 0, 1]**2 -
                                   lines[i, 0, 0]**2 - lines[i, 0, 1]**2) < max_distance_squared,
                            np.abs(lines[:, 0, 2]**2 + lines[:, 0, 3]**2 -
                                   lines[i, 0, 2]**2 - lines[i, 0, 3]**2) < max_distance_squared
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
                    extrems = lines[np.argmax(
                        abs(lines[mask, 0, (1-axis) + 2]-lines[mask, 0, (1-axis)]))][0]
                    # fait la moyenne des coordonnées des lignes sélectionnées selon l'axe
                    med = np.mean(lines[mask, 0, axis])
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

    def draw_lines(self, lines):
        img = np.zeros(
            (self.size_area[0]+50, self.size_area[1]+50, 1), np.uint8)
        if lines is None:
            return img
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(img, (int(x1), int(y1)), (int(x2),
                                              int(y2)), (255, 255, 255), 2)
        # on affiche la position du drone
        x = self.true_position()[0] + self.size_area[0]/2+25
        y = -self.true_position()[1] + self.size_area[1]/2+25
        cv.circle(img, (int(x), int(y)), 5, (255, 255, 255), -1)
        return img

    def control(self):
        command = {"forward": 0.0,
                   "lateral": 0.0, }

        # affiche en temps réel l'ensemble des lignes détectées
        # on affiche aussi la position du drone
        absolute_points = self.convert_lidar_points_to_absolute()
        points = self.convert_absolute_points_to_image_points(absolute_points)
        img_points = self.plot_points(points)

        # première passe, détection des lignes
        base_lines = self.detect_lines_from_points(points)
        img_0 = self.draw_lines(base_lines)

        # seconde passe, permet de transformer les petits bouts de lignes en grands
        second_lines = self.detect_lines_from_img(img_0)
        img_1 = self.draw_lines(second_lines)

        lines_h, lines_v = self.align_lines_on_axis(second_lines)

        lines_h = self.fuse_close_lines(lines_h, 1)
        lines_v = self.fuse_close_lines(lines_v, 0)

        if lines_h is not None and lines_v is not None:
            fused_lines = np.concatenate((lines_h, lines_v))
        elif lines_h is not None:
            fused_lines = lines_h
        elif lines_v is not None:
            fused_lines = lines_v
        else:
            fused_lines = None
        img_2 = self.draw_lines(fused_lines)

        # rounded_lines = self.int_coordonates(aligned_lines)
        rounded_lines = self.int_coordonates(fused_lines)
        img_3 = self.draw_lines(rounded_lines)

        img_affichee = cv.hconcat([img_points, img_0, img_1, img_3])
        cv.imshow("steps", img_affichee)
        cv.waitKey(1)

        # on update les constantes
        for c in self.constants:
            self.constants[c] = cv.getTrackbarPos(c, 'constants')

        self.iter += 1

        return command


class MyMapKeyboard(MapAbstract):

    def __init__(self):
        super().__init__()

        # PARAMETERS MAP
        self._size_area = (400, 400)

        self._number_drones = 1
        self._drones_pos = [((0, 0), 0)]
        self._drones = []

    def construct_playground(self, drone_type: Type[DroneAbstract]):
        playground = ClosedPlayground(size=self._size_area)

        misc_data = MiscData(size_area=self._size_area,
                             number_drones=self._number_drones)
        for i in range(self._number_drones):
            drone = drone_type(identifier=i, misc_data=misc_data)
            self._drones.append(drone)
            playground.add(drone, self._drones_pos[i])

        return playground


def main():
    my_map = MyMapKeyboard()

    playground = my_map.construct_playground(drone_type=MyTestDrone)

    # draw_lidar_rays : enable the visualization of the lidar rays
    # draw_semantic_rays : enable the visualization of the semantic rays
    gui = GuiSR(playground=playground,
                the_map=my_map,
                draw_lidar_rays=True,
                draw_semantic_rays=True,
                use_keyboard=True,
                )
    gui.run()


if __name__ == '__main__':
    main()
