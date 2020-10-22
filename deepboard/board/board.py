from typing import Tuple
import numpy as np
from deepboard.corners.order_corners import OrderCorners


class Board(object):

    def __init__(self, corners: np.ndarray, markers: np.ndarray = None):
        self.corners = OrderCorners.order_points(corners)
        self.markers = OrderCorners.order_bboxes(markers) if markers is not None else None

    @property
    def size(self) -> Tuple[float, float, float, float]:
        """ Length of the 4 sides of the board

        :return: [description]
        :rtype: Tuple[float, float, float, float]
        """

        # o---a---o
        # |       |
        # d       b
        # |       |
        # o---c---o

        a = np.linalg.norm(self.corners[0, :] - self.corners[1, :])
        b = np.linalg.norm(self.corners[1, :] - self.corners[2, :])
        c = np.linalg.norm(self.corners[2, :] - self.corners[3, :])
        d = np.linalg.norm(self.corners[3, :] - self.corners[0, :])

        return a, b, c, d
