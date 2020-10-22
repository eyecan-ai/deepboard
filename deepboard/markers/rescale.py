from typing import Tuple
import numpy as np


class RescaleBoard(object):

    @classmethod
    def rescale_with_markers(cls, markers: np.array) -> Tuple[float]:
        """ Compute the scale ratios of the board size based on the
        markers size

        :param markers: sorted bboxes of the markers [4 x 4]
        :type markers: np.array
        :return: scale factors of board
        :rtype: Tuple[float]
        """

        # from left to right
        # from top to bottom

        ratio_a_w = markers[0, 2] / markers[1, 2]
        ratio_b_h = markers[1, 3] / markers[2, 3]
        ratio_c_w = markers[3, 2] / markers[2, 2]
        ratio_d_h = markers[0, 3] / markers[3, 3]

        return ratio_a_w, ratio_b_h, ratio_c_w, ratio_d_h
