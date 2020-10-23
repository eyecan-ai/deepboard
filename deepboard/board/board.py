from typing import Tuple, Sequence
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

    def rectified_ratio(self, img_size: Sequence[int]) -> float:
        """ Returns the ratio w / h of the rectified board

        Source: https://www.microsoft.com/en-us/research/uploads/prod/2016/11/Digital-Signal-Processing.pdf

        :param img_size: size [h, w] of the original image
        :type img_size: Sequence[int]
        :return: ratio w / h of the rectified board
        :rtype: float
        """

        u0, v0 = img_size[1] / 2, img_size[0] / 2

        m1 = np.append(self.corners[3], 1)
        m2 = np.append(self.corners[2], 1)
        m3 = np.append(self.corners[0], 1)
        m4 = np.append(self.corners[1], 1)

        k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
        k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)

        n2 = k2 * m2 - m1
        n3 = k3 * m3 - m1

        # # handle singularities
        # if k2 == 1 or k3 == 1:
        #     ratio = np.sqrt((n2[0] ** 2 + n2[1] ** 2) / (n3[0] ** 2 + n3[1] ** 2))
        #     return ratio.item()

        f_num1 = n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * u0 ** 2
        f_num2 = n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0 ** 2
        f_den = n2[2] * n3[2]

        # minus in f ???

        f = np.sqrt(np.abs((f_num1 + f_num2) / f_den))

        A = np.array([[f, 0, u0],
                      [0, f, v0],
                      [0, 0, 1]])

        Ati = np.linalg.inv(A.T)
        Ai = np.linalg.inv(A)

        ratio = np.sqrt(np.linalg.multi_dot((n2.T, Ati, Ai, n2)) / np.linalg.multi_dot((n3.T, Ati, Ai, n3)))

        return ratio.item()
