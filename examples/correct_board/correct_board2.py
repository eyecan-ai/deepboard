import math
import click
import cv2
import numpy as np
from deepboard.board.board import Board
from deepboard.corners.order_corners import OrderCorners
from deepboard.markers.rescale import RescaleBoard
from scipy import spatial


def wh_ratio(img, corners):
    h, w = img.shape[:2]
    u0, v0 = w / 2, h / 2

    x1 = corners[3][0]
    y1 = corners[3][1]
    x2 = corners[2][0]
    y2 = corners[2][1]
    x3 = corners[0][0]
    y3 = corners[0][1]
    x4 = corners[1][0]
    y4 = corners[1][1]

    x1 = x1 - u0
    y1 = y1 - v0
    x2 = x2 - u0
    y2 = y2 - v0
    x3 = x3 - u0
    y3 = y3 - v0
    x4 = x4 - u0
    y4 = y4 - v0

    k2 = ((y1 - y4) * x3 - (x1 - x4) * y3 + x1 * y4 - y1 * x4) / ((y2 - y4) * x3 - (x2 - x4) * y3 + x2 * y4 - y2 * x4)
    k3 = ((y1 - y4) * x2 - (x1 - x4) * y2 + x1 * y4 - y1 * x4) / ((y3 - y4) * x2 - (x3 - x4) * y2 + x3 * y4 - y3 * x4)
    f_squared = -((k3 * y3 - y1) * (k2 * y2 - y1) + (k3 * x3 - x1) * (k2 * x2 - x1)) / ((k3 - 1) * (k2 - 1))
    print(k2, k3, f_squared)
    hw_ratio = math.sqrt((pow(k2 - 1, 2) + pow(k2 * y2 - y1, 2) / f_squared + pow(k2 * x2 - x1, 2) / f_squared) / (pow(k3 - 1, 2) + pow(k3 * y3 - y1, 2) / f_squared + pow(k3 * x3 - x1, 2) / f_squared))

    # hw_ratio2 = math.sqrt((pow(y2 - y1, 2) + pow(x2 - x1, 2)) / (pow(y3 - y1, 2) + pow(x3 - x1, 2)))

    return 1 / hw_ratio


def get_corners(event, x, y, flags, params):

    if event == cv2.EVENT_LBUTTONDOWN:
        # params = (corners, img)
        cv2.circle(params[1], (x, y), 5, (0, 0, 255), -1)
        params[0].append([x, y])


def wh_ratio2(img, corners):
    (rows,cols,_) = img.shape

    #image center
    u0 = (cols)/2.0
    v0 = (rows)/2.0

    x1 = corners[3][0]
    y1 = corners[3][1]
    x2 = corners[2][0]
    y2 = corners[2][1]
    x3 = corners[0][0]
    y3 = corners[0][1]
    x4 = corners[1][0]
    y4 = corners[1][1]

    #detected corners on the original image
    p = []
    p.append((x1,y1))
    p.append((x2,y2))
    p.append((x3,y3))
    p.append((x4,y4))

    #widths and heights of the projected image
    w1 = spatial.distance.euclidean(p[0],p[1])
    w2 = spatial.distance.euclidean(p[2],p[3])

    h1 = spatial.distance.euclidean(p[0],p[2])
    h2 = spatial.distance.euclidean(p[1],p[3])

    w = max(w1,w2)
    h = max(h1,h2)

    #visible aspect ratio
    ar_vis = float(w)/float(h)

    #make numpy arrays and append 1 for linear algebra
    m1 = np.array((p[0][0],p[0][1],1)).astype('float32')
    m2 = np.array((p[1][0],p[1][1],1)).astype('float32')
    m3 = np.array((p[2][0],p[2][1],1)).astype('float32')
    m4 = np.array((p[3][0],p[3][1],1)).astype('float32')

    #calculate the focal disrance
    k2 = np.dot(np.cross(m1,m4),m3) / np.dot(np.cross(m2,m4),m3)
    k3 = np.dot(np.cross(m1,m4),m2) / np.dot(np.cross(m3,m4),m2)

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    n21 = n2[0]
    n22 = n2[1]
    n23 = n2[2]

    n31 = n3[0]
    n32 = n3[1]
    n33 = n3[2]

    f = math.sqrt(np.abs( (1.0/(n23*n33)) * ((n21*n31 - (n21*n33 + n23*n31)*u0 + n23*n33*u0*u0) + (n22*n32 - (n22*n33+n23*n32)*v0 + n23*n33*v0*v0))))

    A = np.array([[f,0,u0],[0,f,v0],[0,0,1]]).astype('float32')

    At = np.transpose(A)
    Ati = np.linalg.inv(At)
    Ai = np.linalg.inv(A)

    #calculate the real aspect ratio
    ar_real = math.sqrt(np.dot(np.dot(np.dot(n2,Ati),Ai),n2)/np.dot(np.dot(np.dot(n3,Ati),Ai),n3))

    if ar_real < ar_vis:
        W = int(w)
        H = int(W / ar_real)
    else:
        H = int(h)
        W = int(ar_real * H)

    return ar_real


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

    # ratio = wh_ratio(img, corners)
    # print('ratio', ratio)
    ratio = wh_ratio2(img, corners)
    print('ratio', ratio)

    if markers is not None:
        markers = np.loadtxt(markers)[:, 1:] * [w, h, w, h]
        board = Board(np.array(corners), markers)
    else:
        board = Board(np.array(corners))

    # compute new board size
    a, b, c, d = board.size
    # print(a, b, c, d)
    r_experimental = (a + c) * (b + d) / (4 * b * d)
    board_w_mean = (a + c) / 2
    board_h_mean = (b + d) / 2
    # print(r_experimental, a / d, board_w_mean / board_h_mean, (a*c)/(b*d))
    # r = a / d
    r = ratio
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
        r_a, r_b, r_c, r_d = RescaleBoard.rescale_with_markers(board.markers)
        a_new = a * r_a
        b_new = b * r_b
        c_new = c * r_c
        d_new = d * r_d
        # print(r_a, r_b, r_c, r_d)
        # print(a_new, b_new, c_new, d_new)

        r1 = (r_b + r_d) / 2
        r2 = (r_a + r_c) / 2

        a_new = a * r1
        b_new = b * r2
        c_new = c * r1
        d_new = d * r2
        # print(r1, r2)
        # print(a_new, b_new, c_new, d_new)

        # # a = a * r_w if a >= 1 else a / r_w
        # # d = d * r_h if d >= 1 else d / r_h
        # r = a / d
        # print(r, r_h, r_w)
        # output_height = int(output_width / r)
        # output_corners = np.array([[0, 0],
        #                            [output_width - 1, 0],
        #                            [output_width - 1, output_height - 1],
        #                            [0, output_height - 1]])

        # # compute homography and create the rectangle board
        # homography, _ = cv2.findHomography(board.corners, output_corners)
        # rectangle_board = cv2.warpPerspective(img, homography, (output_width, output_height))

        # cv2.imshow('rectangle board 2', rectangle_board)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
