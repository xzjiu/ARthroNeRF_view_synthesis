import json
import math
import argparse
import os
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Camera Calibration')
parser.add_argument('--img_path', type=str,
                    help='images for calibration')
parser.add_argument('--out_file', type=str, default="transforms.json",
                    help='output path for json file')
parser.add_argument('--width', type=int, default=10,
                    help='the number of the vertices in checkerboard col')
parser.add_argument('--height', type=int, default=7,
                    help='the number of the vertices in checkerboard row')
parser.add_argument('--size', type=float, default=1.0,
                    help='the square_size of each box in (mm)')
args = parser.parse_args()


def CalibrateCamera(path, width, height, square_size):
    images = os.listdir(path)

    objectpts = []
    imagepts = []

    # 3D points real world coordinates
    object3d = np.zeros((width * height, 3), np.float32)
    object3d[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2) * square_size

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    counter = 0

    for image in images:
        img = cv2.imread(path + "/" + image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        counter += 1
        if counter == 400:
            break

        # Find the chess board corners
        # If desired number of corners are
        # found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display
        # them on the images of checkerboard
        if ret:
            objectpts.append(object3d)
            # Refining pixel coordinates
            # for given 2d points.
            corners = cv2.cornerSubPix(gray, np.float32(corners), (11, 11), (-1, -1), criteria)
            imagepts.append(corners)

            # img = cv2.drawChessboardCorners(img, (width, height), corners, ret)
            #
            # cv2.imshow(image, img)
            # cv2.waitKey(0)
    print(len(objectpts))

    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(objectpts, imagepts, gray.shape[::-1], None, None)

    print("RMS: ", ret)

    mean_error = 0

    for i in range(len(objectpts)):
        imagepts2, _ = cv2.projectPoints(objectpts[i], r_vecs[i], t_vecs[i], matrix, distortion)
        error = cv2.norm(imagepts[i], imagepts2, cv2.NORM_L2) / len(imagepts2)
        mean_error += error

    print("Re-projection Error: ", mean_error / len(objectpts))
    print('K:', matrix)
    print('Distorsion: ', distortion)

    return matrix, distortion


# TODO: How to get camera mode, is it must be fisheye?
def TransToJson(matrix, distortion, imgH, imgW, aabb_scale, is_fisheye, outpath='transforms.json'):
    AABB_SCALE = aabb_scale
    h = imgH
    w = imgW
    fl_x = matrix[0][0]
    fl_y = matrix[1][1]
    k1 = distortion[0][0]
    k2 = distortion[0][1]
    k3 = distortion[0][4]
    k4 = distortion[0][5] if len(distortion[0]) >= 6 else 0
    p1 = distortion[0][2]
    p2 = distortion[0][3]
    cx = matrix[0][2]
    cy = matrix[1][2]
    angle_x = math.atan(w / (fl_x * 2)) * 2
    angle_y = math.atan(h / (fl_y * 2)) * 2
    fovx = angle_x * 180 / math.pi
    fovy = angle_y * 180 / math.pi

    out = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "k4": k4,
        "p1": p1,
        "p2": p2,
        "is_fisheye": is_fisheye,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "aabb_scale": AABB_SCALE,
        "frames": [],
    }

    with open(outpath, "w") as outfile:
        json.dump(out, outfile, indent=2)


images = os.listdir(args.img_path)
img1 = cv2.imread(args.img_path + '/' + images[0])
size = img1.shape
w = size[1]
h = size[0]


matrix, distortion = CalibrateCamera(args.img_path, args.width, args.height, args.size)

TransToJson(matrix=matrix, distortion=distortion, imgH=h, imgW=w, aabb_scale=2, is_fisheye=False,
            outpath=args.out_file)
