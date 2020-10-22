import click
import cv2
import numpy as np
from deepboard.board.board import Board
from deepboard.corners.order_corners import OrderCorners
from deepboard.markers.rescale import RescaleBoard


def get_corners(event, x, y, flags, params):

    if event == cv2.EVENT_LBUTTONDOWN:
        # params = (corners, img)
        cv2.circle(params[1], (x, y), 5, (0, 0, 255), -1)
        params[0].append([x, y])


@click.command("")
@click.option("--img", required=True, help="board image file")
@click.option("--markers", required=False, default=None, help="markers bboxes file in yolo format")
@click.option("--output_width", required=True, type=int, help="ouput width")
def run(img, output_width, markers):
    img = cv2.imread(img)
    h, w = img.shape[:2]

    # draw and save the 4 corners
    corners = []
    img_copy = np.copy(img)
    cv2.imshow('board', img_copy)
    cv2.setMouseCallback('board', get_corners, param=(corners, img_copy))
    while len(corners) < 4:
        cv2.imshow('board', img_copy)
        cv2.waitKey(1)

    if markers is not None:
        markers = np.loadtxt(markers)[:, 1:] * [w, h, w, h]
        board = Board(np.array(corners), markers)
    else:
        board = Board(np.array(corners))

    # compute new board size
    a, b, c, d = board.size
    r = a / d
    print(r)
    output_height = int(output_width / r)
    output_corners = np.array([[0, 0],
                               [output_width - 1, 0],
                               [output_width - 1, output_height - 1],
                               [0, output_height - 1]])

    # compute homography and create the rectangle board
    homography, _ = cv2.findHomography(board.corners, output_corners)
    rectangle_board = cv2.warpPerspective(img, homography, (output_width, output_height))

    cv2.imshow('rectangle board', rectangle_board)

    if markers is not None:
        # compute new board size
        r_w, _, _, r_h = RescaleBoard.rescale_with_markers(board.markers)
        a *= r_w
        d *= r_h
        r = a / d
        print(r)
        output_height = int(output_width / r)
        output_corners = np.array([[0, 0],
                                   [output_width - 1, 0],
                                   [output_width - 1, output_height - 1],
                                   [0, output_height - 1]])

        # compute homography and create the rectangle board
        homography, _ = cv2.findHomography(board.corners, output_corners)
        rectangle_board = cv2.warpPerspective(img, homography, (output_width, output_height))

        cv2.imshow('rectangle board 2', rectangle_board)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
