# Calibrate the tool
# Input: a set of image(checkerboard) with corresponding tracking information
# Output: 1. intrinsic matrix in json (intrinsic part for transforms.json)
#         2. Transformation matrix from camera to tracking center
# Run Example: python camera_calibration_hand_eye.py --img_path ./images_handeye/selected/
#            --pose_file pivot_coordinate.csv --camera_out transforms.json --transforms_out transforms_matrix.csv

import json
import math
import csv
import os
import cv2
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser(description='Camera Calibration with Hand Eye')
parser.add_argument('--img_path', type=str, default='./images/selected',
                    help='images for calibration')
parser.add_argument('--pose_file', type=str, default="pivot.csv",
                    help='robot poses with corresponding image path')
parser.add_argument('--camera_out', type=str, default="transforms.json",
                    help='output path for json file')
parser.add_argument('--transforms_out', type=str, default="transforms_matrix.csv",
                    help='output path for camera to tracking center')
parser.add_argument('--width', type=int, default=12,
                    help='the number of the vertices in checkerboard col')
parser.add_argument('--height', type=int, default=8,
                    help='the number of the vertices in checkerboard row')
parser.add_argument('--size', type=float, default=25,
                    help='the square_size of each box in (mm)')
args = parser.parse_args()

# Get relation between image and tracking matrix ################

def stream_read(file):
    stream_dict = []
    with open(file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            stream_dict.append(row)
    return stream_dict


def get_matrix(string):
    transform = string.strip().replace('[', '').replace(']', '')
    rows = transform.split('\n')
    transform_matrix = [list(map(float, row.split())) for row in rows]
    return transform_matrix


def data_process(file):
    stream_dict = stream_read(file)
    # image2tracking is used to
    image2tracking = {}
    # use first frame's camera position as identity matrix
    origin = get_matrix(stream_dict[0]['Tracking'])
    for stream in stream_dict:
        image_path = stream["Image"]
        temp_matrix = stream["Tracking"]
        if re.search("nan", temp_matrix):
            if os.path.exists(image_path):
                os.remove(image_path)
        else:
            tracking_matrix = get_matrix(temp_matrix)
            store_image = image_path.split('/')[-1]
            image2tracking[store_image] = np.dot(np.linalg.inv(origin), tracking_matrix)
    return image2tracking


# Hand Eye Calibration
def CalibrateHandEye(path, width, height, square_size, robo_map):
    images = os.listdir(path)

    robot_poses = []
    objectpts = []
    imagepts = []

    # 3D points real world coordinates
    object3d = np.zeros((width * height, 3), np.float32)
    object3d[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2) * square_size

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    img1 = cv2.imread(path + '/' + images[0])
    size = img1.shape
    w = size[1]
    h = size[0]

    for image in images:
        img = cv2.imread(path + "/" + image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        # If desired number of corners are
        # found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display
        # them on the images of checkerboard
        if ret:
            robot_poses.append(robo_map[image])
            objectpts.append(object3d)
            # Refining pixel coordinates
            # for given 2d points.
            corners = cv2.cornerSubPix(gray, np.float32(corners), (11, 11), (-1, -1), criteria)
            imagepts.append(corners)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(objectpts, imagepts, gray.shape[::-1], None, None)

    R_all_end_to_base_1 = []
    T_all_end_to_base_1 = []

    for robot_pose in robot_poses:
        print(robot_pose)
        R_all_end_to_base_1.append(robot_pose[:3, :3])
        T_all_end_to_base_1.append(robot_pose[:3, 3].reshape(3, 1))

    R, T = cv2.calibrateHandEye(R_all_end_to_base_1, T_all_end_to_base_1, r_vecs, t_vecs)
    print("Camera regard to End:\n")
    print("Rotation: ", R)
    print("Translation: ", T)
    RT = np.column_stack((R, T))
    RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
    print("Transformation_matrix: ", RT)
    return RT, matrix, distortion, w, h


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


# Get dict of image2dict

image_map = data_process(args.pose_file)
RT, matrix, distortion, w, h = CalibrateHandEye(args.img_path, args.height, args.width, args.size, image_map)

# Write and save image information
with open(args.transforms_out, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(RT)

TransToJson(matrix=matrix, distortion=distortion, imgH=h, imgW=w, aabb_scale=4, is_fisheye=False,
            outpath=args.camera_out)
