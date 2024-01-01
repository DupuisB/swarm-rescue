"""
Graph implementation, here are the characteristics of the graph:
    - Each node is associated with a unique ID, a position (x, y), a isImportant boolean, a list of neighbors
"""

import cv2
import arcade
from typing import Union
from spg_overlay.entities.drone_abstract import DroneAbstract

class Node:
    def __init__(self, x:Union[int, float, tuple], y=None, neighbors=None, primary=False, N=0, E=0, S=0, W=0):

        self.id = id(self)

        if neighbors is None:
            neighbors = []
        if y:
            self.x = x # world coordinates
            self.y = y
        else:
            self.x = x[0]
            self.y = x[1]
        self.neighbors: list = neighbors
        self.isVisible = False
        self.isGap = False
        self.isPrimary: bool = primary # Corners and gaps are primary
        self.weight: int = 1 #Number of times the node has been seen
        self.directions = {"N": N, "E": E, "S": S, "W": W} # -1 wall, 0 unknown, 1 open and unexplored, 2 open and being explored, 3 open and explored
        self.timer = 50 #Number of frames before a node is considered old (and removed)

    def __eq__(self, other, threshold=20):
        """
        If the coordinates are close enough
        """
        if self.isGap != other.isGap or self.isPrimary != other.isPrimary:
            return False
        self.timer += 50
        return abs(self.x - other.x) < threshold and abs(self.y - other.y) < threshold

    def update(self):
        if self.timer > 0:
            self.timer -= 1
        elif self.isPrimary:
            self.weight = 0
        else:
            pass

class Graph:
    def __init__(self):
        self.nodes = []
        self.visited = {}
        self.edges = []
        self.corners = []
        self.gaps = []

    def update(self):
        for node in self.nodes:
            node.update()

    def add_node(self, node: Node, position=None, sec:Node=None):
        # sec is Last Secondary
        # position is the position of the drone
        # Check if the node is already in the graph using __eq__
        for n in self.nodes:
            if n == node:
                # Averages the position of the two nodes, weighted by the number of times they have been seen
                n.x = int((n.x*n.weight + node.x)/(n.weight+1))
                n.y = int((n.y*n.weight + node.y)/(n.weight+1))
                n.weight += 1
                if n.weight == 2 and position is not None:
                    new_node = Node(position[0], position[1], [n], primary=False)
                    self.nodes.append(new_node)
                    self.visited[new_node.id] = False
                    n.neighbors.append(new_node)
                    if sec is not None:
                        sec.neighbors.append(new_node)
                        new_node.neighbors.append(sec)
                return
        self.nodes.append(node)
        self.visited[node.id] = False

    def delete_node(self, node: Node):
        for i in range(len(self.nodes)):
            if self.nodes[i] == node:
                del self.nodes[i]
            else:
                for j in range(len(self.nodes[i].neighbors)):
                    if self.nodes[i].neighbors[j] == node:
                        del self.nodes[i].neighbors[j]

    def draw_arcade(self):
        """
        Draw the graph on the image
        """
        for node in self.nodes:
            if node.weight == 0:
                continue
            if node.isPrimary:
                if node.isVisible:
                    arcade.draw_circle_filled(node.x, node.y, 10, (255, 255, 255))
                    node.isVisible = False
                elif node.weight >= 3: #TODO: check the >= 3 everywhere
                    arcade.draw_circle_filled(node.x, node.y, 10, (0, 0, 255))
                else:
                    pass
            else:
                arcade.draw_circle_filled(node.x, node.y, 5, (0, 255, 0))
            for neighbor in node.neighbors:
                if neighbor is None:
                    continue
                if neighbor.isPrimary and node.isPrimary and node.weight >= 3 and neighbor.weight >= 3:
                    #Draw a pink line
                    arcade.draw_line(node.x, node.y, neighbor.x, neighbor.y, (255, 0, 255), 2)
                else:
                    #arcade.draw_line(node.x, node.y, neighbor.x, neighbor.y, (0, 255, 0), 2)
                    pass