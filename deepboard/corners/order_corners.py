import numpy as np


class OrderCorners(object):

    @classmethod
    def order_points(cls, points: np.ndarray) -> np.ndarray:
        """ Sort a list of 4 points (x, y) such that the first entry
        in the list is the top-left, the second entry is the top-right,
        the third is the bottom-right, and the fourth is the bottom-left

        :param points: input points [4 x 2]
        :type points: np.ndarray
        :return: ordered points [4 x 2]
        :rtype: np.ndarray
        """

        rect = np.zeros((4, 2), dtype=np.float32)

        # compute the sum, the top-left will have the smallest
        # sum, whereas the bottom-right will have the largest sum
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]

        # compute the difference, the top-right will have the smallest
        # difference, whereas the bottom-left will have the largest difference
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]

        return rect

    @classmethod
    def order_bboxes(cls, bboxes: np.ndarray) -> np.ndarray:
        """ Sort a list of 4 bboxes (x, y, w, h) such that the first entry
        in the list is the top-left, the second entry is the top-right,
        the third is the bottom-right, and the fourth is the bottom-left

        :param bboxes: input bboxes [4 x 4]
        :type bboxes: np.ndarray
        :return: sorted bboxes [4 x 4]
        :rtype: np.ndarray
        """

        rect = np.zeros((4, 4), dtype=np.float32)

        # compute the sum, the top-left will have the smallest
        # sum, whereas the bottom-right will have the largest sum
        s = bboxes[:, :2].sum(axis=1)
        rect[0] = bboxes[np.argmin(s)]
        rect[2] = bboxes[np.argmax(s)]

        # compute the difference, the top-right will have the smallest
        # difference, whereas the bottom-left will have the largest difference
        diff = np.diff(bboxes[:, :2], axis=1)
        rect[1] = bboxes[np.argmin(diff)]
        rect[3] = bboxes[np.argmax(diff)]

        return rect
