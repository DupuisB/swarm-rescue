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


class MyTestDrone(DroneAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nb_of_iter_counter = 0
        # delta x / delta y par rapport à la position actuelle
        self.position_consigne = [0.0, 100.0]

        self.prev_diff_position = 0

        # initialisation du filtre de kalman

        self.kalman_filter = cv.KalmanFilter(4, 2)
        self.kalman_filter.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)
        self.kalman_filter.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)

        # je sais pas trop ce que fou ce coef ici mais ok
        self.kalman_filter.processNoiseCov = (
            np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], np.float32) * 0.03
        )

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def update_kalman_filters(self):
        m_pos = self.measured_gps_position()
        m = np.array((m_pos[0], m_pos[1]), dtype=np.float32)
        self.kalman_filter.correct(m)

    def predict_gps_pos(self):
        pred = self.kalman_filter.predict()
        return np.array((pred[0][0], pred[1][0]), np.float32)

    def relative_pos_to_abs(self, delta_lat, delta_vert):
        # delta_lat positif => vers la droite du drone
        # delta_vert positif => vers l'avant
        alpha = self.measured_compass_angle()
        delta_x = np.cos(alpha) * delta_vert + np.sin(alpha) * delta_lat
        delta_y = np.sin(alpha) * delta_vert - np.cos(alpha) * delta_lat
        return (self.true_position()[0] + delta_x, self.true_position()[1] + delta_y)

    def abs_pos_to_relative(self, pos_x, pos_y):
        alpha = self.measured_compass_angle()

    def control(self):
        # print(f"Current position consigne : {self.position_consigne}")

        self.update_kalman_filters()

        if self.nb_of_iter_counter % 70 == 0:
            print(
                f"True : {self.true_position()} | GPS : {self.measured_gps_position()} | Kalman : {self.predict_gps_pos()}")
            print(
                f"Delta GPS : {self.true_position() - self.measured_gps_position()} | Delta Kalman : {self.true_position() - self.predict_gps_pos()} ")
            print("\n")
        self.nb_of_iter_counter += 1

        lidar_vals = self.lidar_values()

        command = {"forward": 0.0,
                   "lateral": 0.0,
                   }

        deb = 43
        ed = 47
        estim_values = list(map(lambda x: -np.sin(
            x[1]) * x[0],  zip(lidar_vals[deb:ed], self.lidar_rays_angles()[deb:ed])))

        estim_dist_a_droite = sum(estim_values) / len(estim_values)

        if False:  # abs(estim_dist_a_droite - 50) > 5:  # va se coler au mur à sa droite
            # print("faut droite")
            self.position_consigne[1] = self.relative_pos_to_abs(
                estim_dist_a_droite - 50, 0)[1]

        # autre approche : quand on est assez près de la consigne, on arrète de s'en rapprocher, on bouge plus
        # ça évite d'être soumis aux aléas du gps

        if self.position_consigne is not None:

            diff_position = np.array(
                self.position_consigne) - self.true_position()

            deriv_diff_position = diff_position - self.prev_diff_position
            # PD filter 1
            Ku = 25 / 100  # Gain debut oscillation maintenue en P pure
            Tu = 26  # Période d'oscillation
            Kp = 0.8 * Ku
            Kd = Ku * Tu / 10.0

            lat = clamp(Kp * diff_position[1] +
                        Kd * deriv_diff_position[1], -1.0, 1.0)

            """ x_consigne, y_consigne = map(lambda v: clamp(
                v, -1.0, 1.0), Kp * diff_position + Kd * deriv_diff_position)
              """
            command["lateral"] = lat
            # print("consigne latérale = ", lat)

        else:
            self.prev_diff_position = 0

        return command


class MyMapKeyboard(MapAbstract):

    def __init__(self):
        super().__init__()

        # PARAMETERS MAP
        self._size_area = (600, 600)

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
                use_keyboard=False,
                )
    gui.run()


if __name__ == '__main__':
    main()
