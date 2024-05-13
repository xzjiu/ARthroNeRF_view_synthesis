import json
import math
import csv
import os
import cv2
import numpy as np
import re
import pickle


################ Get relation between image and tracking matrix ################

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
    image2tracking = {}
    for stream in stream_dict:
        image_path = stream["Image"]
        temp_matrix = stream["Tracking"]
        if re.search("nan", temp_matrix):
            if os.path.exists(image_path):
                os.remove(image_path)
        else:
            tracking_matrix = get_matrix(temp_matrix)
            store_image = image_path.split('/')[-1]
            image2tracking[store_image] = np.asarray(tracking_matrix)
    return image2tracking


########################################################################################
K = np.array([[471.32480599,   0.00000000e+00, 319.6930523 ],
            [0.00000000e+00, 473.00882149, 244.39997145],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float64)
distortion = np.array([[ 4.67626869e-03, 6.27888081e-02, 8.77125257e-04, 2.36776524e-04, -2.60426004e-01]])
target_x_number = 12
target_y_number = 7
target_cell_size = 20


def CalibrateHandEye(path, width, height, square_size, robo_map, outputpath):
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
    r_vecs = []
    t_vecs = []
    poses = []
    file_pathes = []

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
            retval, rvec, tvec = cv2.solvePnP(object3d, corners, K, distCoeffs=distortion)
            r_vecs.append(rvec)
            t_vecs.append(tvec)
            poses.append(getHomo(rvec, tvec))
            file_pathes.append(path+"/"+image)

            # test_r_vecs, test_t_vecs = [], []
            # for r_vec, t_vec in zip(r_vecs, t_vecs):
            #     r_vec = cv2.Rodrigues(r_vec)[0]
            #     print(r_vec)
            #     RT_camera = np.column_stack((r_vec, t_vec))
            #     RT_camera = np.row_stack((RT_camera, np.array([0, 0, 0, 1])))
            #     RT_camera = np.linalg.inv(RT_camera)
            #     test_r_vecs.append(RT_camera[:3, :3])
            #     test_t_vecs.append(RT_camera[:3, 3].reshape(3, 1))
            
    
    # ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(objectpts, imagepts, gray.shape[::-1], None, None)
    

    poses_array = np.asarray(poses)

    with open('camera_poses.pkl', 'wb') as f:
        pickle.dump(poses_array, f)

    with open('camera_poses_image_path.pkl', 'wb') as f:
        pickle.dump(file_pathes, f)

    R_all_end_to_base_1 = []
    T_all_end_to_base_1 = []

    for robot_pose in robot_poses:
        # robot_pose = np.linalg.inv(robot_pose)
        R_all_end_to_base_1.append(robot_pose[:3, :3])
        T_all_end_to_base_1.append(robot_pose[:3, 3].reshape(3, 1))

    R, T = cv2.calibrateHandEye(R_all_end_to_base_1, T_all_end_to_base_1, r_vecs, t_vecs)
    print("Camera regard to End:\n")
    print("Rotation: ", R)
    print("Translation: ", T)
    RT = np.column_stack((R, T))
    RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
    print("Transformation_matrix: ", RT)
    print("Inverse_transformation: ", np.linalg.inv(RT))

    # Write and save image information
    with open(outputpath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(RT)
    return RT, distortion, w, h

def getHomo(r_vec, t_vec):
    r_vec = cv2.Rodrigues(r_vec)[0]
    RT_camera = np.column_stack((r_vec, t_vec))
    RT_camera = np.row_stack((RT_camera, np.array([0, 0, 0, 1])))
    return RT_camera


# Get dict of image2dict
image_map = data_process('./camera_calibration/pivot_coordinate.csv')
RT, distortion, w, h = CalibrateHandEye('./camera_calibration/images_handeye/selected/', 12, 8, 20, image_map, "./camera_calibration/handeye_matrix.csv")
