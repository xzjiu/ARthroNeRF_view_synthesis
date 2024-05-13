
import json
import math
import csv
import os
import cv2
import numpy as np
import re
import pickle
from sklearn.model_selection import KFold

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
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners, ret)
            # test_r_vecs, test_t_vecs = [], []
            # for r_vec, t_vec in zip(r_vecs, t_vecs):
            #     r_vec = cv2.Rodrigues(r_vec)[0]
            #     print(r_vec)
            #     RT_camera = np.column_stack((r_vec, t_vec))
            #     RT_camera = np.row_stack((RT_camera, np.array([0, 0, 0, 1])))
            #     RT_camera = np.linalg.inv(RT_camera)
            #     test_r_vecs.append(RT_camera[:3, :3])
            #     test_t_vecs.append(RT_camera[:3, 3].reshape(3, 1))
        # cv2.imshow(image,img)
        # cv2.waitKey(0)    
    
    ret, matrix, distortion_1, r_vecs, t_vecs = cv2.calibrateCamera(objectpts, imagepts, gray.shape[::-1], None, None)

    print("RET: ", ret)
    print("Used intrinsic: ", K)
    print("Actual intrinsic: ", matrix)

    print("\n Used distortion: ", distortion)
    print("Actual distrotion: ", distortion_1)
    

    poses_array = np.asarray(poses)

    # with open('camera_poses.pkl', 'wb') as f:
    #     pickle.dump(poses_array, f)

    # with open('camera_poses_image_path.pkl', 'wb') as f:
    #     pickle.dump(file_pathes, f)

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

    ##############################################################################################################

    def calculate_spatial_error(R_est, T_est, robot_R_matrices, robot_T_vectors, camera_R_vectors, camera_T_vectors):
        """
        Calculate the spatial error by computing the Euclidean distance between the final positions of
        a reference point after applying the predicted and actual transformations.
        """
        errors = []
        # Define a reference point in the robot's coordinate system (e.g., the tip of the end-effector)
        reference_point = np.array([0, 0, 0, 1])  # Homogeneous coordinates for easy matrix multiplication
        
        for R_robot, T_robot, R_camera_vec, T_camera in zip(robot_R_matrices, robot_T_vectors, camera_R_vectors, camera_T_vectors):
            # Convert camera's Rodrigues rotation vector to a rotation matrix for the actual pose
            R_camera_actual, _ = cv2.Rodrigues(R_camera_vec)
            
            # Construct full 4x4 transformation matrices for the predicted and actual poses
            T_pred = np.vstack((np.hstack((R_est @ R_robot, R_est @ T_robot + T_est)), [0, 0, 0, 1]))
            T_actual = np.vstack((np.hstack((R_camera_actual, T_camera)), [0, 0, 0, 1]))
            
            # Transform the reference point by both the predicted and actual transformations
            point_pred = T_pred @ reference_point
            point_actual = T_actual @ reference_point
            
            # Compute the Euclidean distance between the transformed points' spatial locations
            distance = np.linalg.norm(point_pred[:3] - point_actual[:3])  # Ignore the homogeneous coordinate
            errors.append(distance)
        
        # Compute the average of the spatial errors
        average_error = np.mean(errors)
        return average_error

    def calculate_combined_reprojection_error(R_est, T_est, robot_R_matrices, robot_T_vectors, camera_R_vectors, camera_T_vectors):
        """
        Calculate a combined reprojection error by comparing the final positions of a reference point
        after applying the predicted and actual transformations.
        """
        errors = []
        # Define a reference point in the robot's coordinate system (e.g., the tip of the end-effector)
        reference_point = np.array([0, 0, 0, 1])  # Homogeneous coordinates
        
        for R_robot, T_robot, R_camera_vec, T_camera in zip(robot_R_matrices, robot_T_vectors, camera_R_vectors, camera_T_vectors):
            # Convert camera's Rodrigues rotation vector to a rotation matrix
            R_camera_actual, _ = cv2.Rodrigues(R_camera_vec)
            
            # Construct the full transformation matrices (4x4) for predicted and actual camera poses
            T_pred = np.vstack((np.hstack((R_est @ R_robot, R_est @ T_robot + T_est)), [0, 0, 0, 1]))
            T_actual = np.vstack((np.hstack((R_camera_actual, T_camera)), [0, 0, 0, 1]))
            
            # Apply transformations to the reference point
            point_pred = T_pred @ reference_point
            point_actual = T_actual @ reference_point
            
            # Calculate the error as the Euclidean distance between the transformed points
            error = np.linalg.norm(point_pred[:3] - point_actual[:3])  # Ignore the homogeneous coordinate for distance calculation
            errors.append(error)
        
        # Average error across all points
        average_error = np.mean(errors)
        return average_error
    

    def calculate_reprojection_error(R_est, T_est, robot_R_matrices, robot_T_vectors, camera_R_vectors, camera_T_vectors):
        """
        Calculate the reprojection error given the hand-eye calibration result (R_est, T_est),
        lists of test robot poses (rotation matrices and translation vectors),
        and the corresponding camera poses (Rodrigues rotation vectors and translation vectors).
        
        Parameters:
        - R_est, T_est: Estimated rotation matrix and translation vector from hand-eye calibration.
        - robot_R_matrices: List of robot rotation matrices.
        - robot_T_vectors: List of robot translation vectors.
        - camera_R_vectors: List of camera Rodrigues rotation vectors.
        - camera_T_vectors: List of camera translation vectors.
        
        Returns:
        - Average translation and rotation error.
        """
        translation_errors, rotation_errors = [], []
        for R_robot, T_robot, R_camera_vec, T_camera in zip(robot_R_matrices, robot_T_vectors, camera_R_vectors, camera_T_vectors):
            # Convert camera's Rodrigues rotation vector to a rotation matrix
            
            R_camera_actual, _ = cv2.Rodrigues(R_camera_vec)

            # Predict the camera pose using the estimated hand-eye transformation
            R_camera_pred = R_est @ R_robot
            T_camera_pred = R_est @ T_robot + T_est

            # Calculate translation error as Euclidean distance
            print("T_camera:",T_camera)
            print("T_camera_pred", T_camera_pred)
            translation_error = np.linalg.norm(T_camera - T_camera_pred)
            translation_errors.append(translation_error)

            # Calculate rotation error as the angle between actual and predicted orientations
            R_diff = R_camera_pred.T @ R_camera_actual
            angle = cv2.Rodrigues(R_diff)[0]  # Convert back to Rodrigues vector to get the rotation angle
            rotation_error = np.linalg.norm(angle)  # Angle magnitude represents rotation error
            rotation_errors.append(rotation_error)
            
        average_translation_error = np.mean(translation_errors)
        average_rotation_error = np.mean(rotation_errors)
        return average_translation_error, average_rotation_error


    # Number of splits
    k = 9
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # To store the errors for each fold
    R_errors, T_errors = [], []
    errors = []

    for train_index, test_index in kf.split(robot_poses):
        # Splitting the dataset into training and testing
        R_all_end_to_base_1_train, R_all_end_to_base_1_test = np.array(R_all_end_to_base_1)[train_index], np.array(R_all_end_to_base_1)[test_index]
        T_all_end_to_base_1_train, T_all_end_to_base_1_test = np.array(T_all_end_to_base_1)[train_index], np.array(T_all_end_to_base_1)[test_index]
        r_vecs_train, r_vecs_test = np.array(r_vecs)[train_index], np.array(r_vecs)[test_index]
        t_vecs_train, t_vecs_test = np.array(t_vecs)[train_index], np.array(t_vecs)[test_index]
        
        # Perform calibration on the training set
        # Assuming you have already converted poses to the appropriate format for cv2.calibrateHandEye
        R, T = cv2.calibrateHandEye(R_all_end_to_base_1_train, T_all_end_to_base_1_train, r_vecs_train, t_vecs_train)
        print("Rotation: ", R_all_end_to_base_1)
        print("Translation: ", T_all_end_to_base_1_train)
        print("r_vecs: ", r_vecs_train)
        print("t_vecs: ", t_vecs_train)
        
        # Evaluate on the testing set
        # Here you should implement the reprojection error calculation based on your specific requirements
        # For simplicity, we'll just append a placeholder error
        error = calculate_spatial_error(R, T, R_all_end_to_base_1_test, T_all_end_to_base_1_test, r_vecs_test, t_vecs_test)
        T_error, R_error = calculate_reprojection_error(R, T, R_all_end_to_base_1_test, T_all_end_to_base_1_test, r_vecs_test, t_vecs_test)
        errors.append(error)
        R_errors.append(R_error)
        T_errors.append(T_error)

    # Calculate the average error across all folds
    print(np.mean(errors))
    average_R_error = np.mean(R_errors)
    average_T_error = np.mean(T_errors)
    print(f"Average reprojection error across {k} folds: {average_R_error}")
    print(f"Average reprojection error across {k} folds: {average_T_error}")

    ##############################################################################################################

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
image_map = data_process('C:/Users/camp/GIT/arthro_nerf/handeye-v2/pivot.csv')
RT, distortion, w, h = CalibrateHandEye('C:/Users/camp/GIT/arthro_nerf/handeye-v2/images/selected/', 10, 7, 10, image_map, "C:/Users/camp/GIT/arthro_nerf/handeye-v2/handeye_matrix_verified.csv")
