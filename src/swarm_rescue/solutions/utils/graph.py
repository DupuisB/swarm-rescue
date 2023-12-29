"""
Graph implementation, here are the characteristics of the graph:
    - Each node is associated with a unique ID, a position (x, y), a isImportant boolean, a list of neighbors
"""

import cv2
import arcade
from spg_overlay.entities.drone_abstract import DroneAbstract

class Node:
    def __init__(self, x, y, neighbors=None, primary=False, N=0, E=0, S=0, W=0):
        if neighbors is None:
            neighbors = []
        self.x = x
        self.y = y
        self.neighbors = neighbors
        self.isPrimary = primary # Corners and gaps are primary
        self.weight = 1 #Number of times the node has been seen
        self.directions = {"N": N, "E": E, "S": S, "W": W} # -1 wall, 0 unknown, 1 open and unexplored, 2 open and explored
        self.timer = 50 #Number of frames before a node is considered old (and removed)

    def __eq__(self, other, threshold=20):
        """
        If the coordinates are close enough
        """
        self.timer += 50
        return abs(self.x - other.x) < threshold and abs(self.y - other.y) < threshold

    def update(self):
        if self.timer > 0:
            self.timer -= 1
        else:
            self.weight = 0

class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.corners = []
        self.gaps = []

    def update(self):
        for node in self.nodes:
            node.update()

    def add_node(self, node):
        # Check if the node is already in the graph using __eq__
        for n in self.nodes:
            if n == node:
                # Averages the position of the two nodes, weighted by the number of times they have been seen
                n.x = int((n.x*n.weight + node.x)/(n.weight+1))
                n.y = int((n.y*n.weight + node.y)/(n.weight+1))
                return
        self.nodes.append(node)

    def draw(self, img):
        """
        Draw the graph on the image
        """
        for node in self.nodes:
            if node.weight == 0:
                continue
            if node.isPrimary:
                cv2.circle(img, (node.x, node.y), 10, (0, 0, 255), -1)
            else:
                cv2.circle(img, (node.x, node.y), 5, (0, 255, 0), -1)
            for neighbor in node.neighbors:
                cv2.line(img, (node.x, node.y), (neighbor.x, neighbor.y), (0, 255, 0), 2)


    def draw_arcade(self):
        """
        Draw the graph on the image
        """
        for node in self.nodes:
            if node.weight == 0:
                continue
            if node.isPrimary:
                arcade.draw_circle_filled(node.x, node.y, 10, (0, 0, 255))
            else:
                arcade.draw_circle_filled(node.x, node.y, 5, (0, 255, 0))
            for neighbor in node.neighbors:
                arcade.draw_line(node.x, node.y, neighbor.x, neighbor.y, (0, 255, 0), 2)