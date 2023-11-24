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


class MyTestDrone(DroneAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nb_of_iter_counter = 0
        # delta x / delta y par rapport à la position actuelle
        self.position_consigne = [0.0, 100.0]
        self.iter = 0
        self.prev_diff_position = 0

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

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
            self.lidar_values(), self.lidar_rays_angles()) if d < 280])
        # récupère les tableaux des distances et des angles correspondant
        lidar_distance = np.array(list(map(lambda x: x[0], lidar_data)))
        lidar_angles = np.array(list(map(lambda x: x[1], lidar_data)))
        # print(lidar_distance)
        # print(lidar_angles)
        lidar_x = []
        lidar_y = []
        for i in range(len(lidar_distance)-1):
            lidar_x.append(
                lidar_distance[i] * np.cos(lidar_angles[i]) + self.measured_gps_position()[0] + taille_carte[0]/2+25)
            lidar_y.append(
                lidar_distance[i] * np.sin(lidar_angles[i]) + self.measured_gps_position()[1] + taille_carte[1]/2+25)
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
        cv.imwrite('points.png', img)

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

    def align_lines_on_axis(self, lines):
        # aligne les lignes en considérant qu'elles sont soit horizontales, soit verticales
        # on considère que les lignes sont horizontales si leur angle est proche de 0 ou de pi
        # on considère que les lignes sont verticales si leur angle est proche de pi/2 ou de 3*pi/2

        # on calcule l'angle de chaque ligne
        angles = np.arctan2(lines[:, 0, 3] - lines[:, 0, 1],
                            lines[:, 0, 2] - lines[:, 0, 0])
        eps = 30*np.pi/180
        # une ligne est horizontale si pi-eps < angle < pi+eps ou -eps < angle < eps
        mask_horizontal = np.logical_or(
            np.logical_and(angles > np.pi-eps, angles < np.pi+eps),
            np.logical_and(angles > -eps, angles < eps)
        )
        # une ligne est verticale si pi/2-eps < angle < pi/2+eps ou 3*pi/2-eps < angle < 3*pi/2+eps
        mask_vertical = np.logical_or(
            np.logical_and(angles > np.pi/2-eps, angles < np.pi/2+eps),
            np.logical_and(angles > 3*np.pi/2-eps, angles < 3*np.pi/2+eps)
        )
        # pour chaque ligne horizontale on calcule son y moyen
        # pour chaque ligne verticale on calcule son x moyen
        lignes_horizontales = lines[mask_horizontal]
        lignes_verticales = lines[mask_vertical]
        # pour chaque ligne horizontale on calcule son y moyen
        y_moyens = np.array(
            [lignes_horizontales[:, 0, 1] + lignes_horizontales[:, 0, 3]])/2
        x_moyens = np.array(
            [lignes_verticales[:, 0, 0] + lignes_verticales[:, 0, 2]])/2
        # poru les lignes horizontales on garde comme coordonnées x celles déjà existantes et on prend comme coordonnées y le y moyen
        lignes_horizontales[:, 0, 1] = y_moyens
        lignes_horizontales[:, 0, 3] = y_moyens
        # pour les lignes verticales on garde comme coordonnées y celles déjà existantes et on prend comme coordonnées x le x moyen
        lignes_verticales[:, 0, 0] = x_moyens
        lignes_verticales[:, 0, 2] = x_moyens
        # on ne garde que les lignes horizontales et verticales
        # on fusionne les lignes horizontales et verticales
        # on retourne les lignes
        return lignes_horizontales, lignes_verticales

    def fuse_close_lines(self, lines, axis):
        # fusionne les lignes (horizontales ou verticales) qui sont proches selon l'axe donné
        # on considère que deux lignes sont proches si la valeur selon l'axe de leur premier point est proche
        # axis vaut 0 pour les verticales et 1 pour les horizontales (les horizontales ont un y constant)
        already_fused = np.zeros(len(lines), dtype=bool)
        new_lines = []
        max_delta = 30
        for i in range(len(lines)):
            if not already_fused[i]:
                # on crée un masque pour ne regarder que les lignes non sélectionnées
                # et qui sont proches de la ligne courante
                # on prend aussi la ligne courante
                mask = np.logical_and(
                    np.logical_not(already_fused),
                    np.abs(lines[:, 0, axis] - lines[i, 0, axis]) < max_delta
                )
                already_fused[mask] = True
                # on fusionne les lignes sélectionnées
                if np.sum(mask) >= 1:
                    new_lines.append(np.mean(lines[mask], axis=0))
        lines = np.array(new_lines)
        # print(lines.shape)
        if lines.shape[0] == 0:
            return lines
        return self.int_coordonates(lines)

    def int_coordonates(self, lines):
        # arrondie les coordonnées des lignes aux entiers
        lines[:, 0, 0] = np.round(lines[:, 0, 0])
        lines[:, 0, 1] = np.round(lines[:, 0, 1])
        lines[:, 0, 2] = np.round(lines[:, 0, 2])
        lines[:, 0, 3] = np.round(lines[:, 0, 3])
        return lines

    def control(self):
        command = {"forward": 0.0,
                   "lateral": 0.0, }

        # affiche en temps réel l'ensemble des lignes détectées
        # on affiche aussi la position du drone
        base_lines = self.detect_murs()
        # print(lines.shape)
        # print(lines[:10])
        # x = len(base_lines)
        # lines = self.fuse_lines(base_lines)
        # if len(lines) < x:
        #    print("Une fusion a fait gagner : ", x-len(lines), " lignes")
        lignes_h, lignes_v = self.align_lines_on_axis(base_lines)
        # print(lignes_h.shape)
        lignes = np.concatenate((lignes_h, lignes_v))
        """ lignes_h_fusees = self.fuse_close_lines(lignes_h, 1)
        lignes_v_fusees = self.fuse_close_lines(lignes_v, 0)

        # si l'un des deux est vide on prend l'autre
        if lignes_h_fusees.shape[0] == 0:
            lignes = lignes_v_fusees
        elif lignes_v_fusees.shape[0] == 0:
            lignes = lignes_h_fusees
        else:
            lignes = np.concatenate((lignes_h_fusees, lignes_v_fusees)) 
        """

        print("Nombre de lignes détectées : ", lignes.shape[0])

        if True:
            lines = self.detect_murs()
            img = np.zeros(
                (self.size_area[0]+50, self.size_area[1]+50, 1), np.uint8)
            for ligne in lignes:
                x1, y1, x2, y2 = ligne[0]
                cv.line(img, (int(x1), int(y1)), (int(x2),
                                                  int(y2)), (255, 255, 255), 2)
            # on affiche la position du drone
            x = self.true_position()[0] + self.size_area[0]/2+25
            y = self.true_position()[1] + self.size_area[1]/2+25
            cv.circle(img, (int(x), int(y)), 5, (255, 255, 255), -1)
            cv.imshow("img", img)
            cv.waitKey(1)

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
