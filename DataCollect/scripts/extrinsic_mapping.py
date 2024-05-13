# Add frames to transforms.json

import json
import math
import csv
import os
import cv2
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser(description='Add extrinsic frames to transforms.json')
parser.add_argument('--pose_file', type=str, default="pivot_carbox.csv",
                    help='robot poses with corresponding image path')
parser.add_argument('--camera_file', type=str, default="transforms.json",
                    help='output path for json file')
parser.add_argument('--transforms_matrix', type=str, default="transforms_matrix.csv",
                    help='camera to tracking center')
args = parser.parse_args()


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def get_sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


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


def process_frames(tracker_streams, camera2robot_file):
    frames = []
    camera2robot = np.loadtxt(camera2robot_file, dtype=float, delimiter=',')
    print(camera2robot)
    if not tracker_streams:
        raise Exception("No data collected!")
    # ensure origin is not nan
    while tracker_streams:
        if not re.search("nan", tracker_streams[0]['Tracking']):
            break
        else:
            tracker_streams.pop(0)

    # use first frame's camera position as identity matrix
    offset = [[ 1, 0, 0, 5 ],
              [ 0, 1, 0, 0 ],
              [ 0, 0, 1, 0],
              [ 0, 0, 0, 1]]
    origin = np.dot(np.linalg.inv(camera2robot), get_matrix(tracker_streams[0]['Tracking'])) # original
    origin = np.dot(offset, origin)

    for stream in tracker_streams:
        file_path = stream['Image']
        sharpness = get_sharpness(file_path)
        transform = stream['Tracking']
        if re.search("nan", transform):
            if os.path.exists(file_path):
                os.remove(file_path)
        else:
            temp_matrix = get_matrix(transform)
            transform_matrix = np.dot(np.linalg.inv(camera2robot), temp_matrix)
            transform_matrix = np.dot(np.linalg.inv(origin), transform_matrix)
            transform_matrix[0:3, 3] *= 0.01
            # transform_matrix = np.dot(np.linalg.inv(camera2robot), camera_matrix)
            # transform_matrix = camera_matrix
            frames.append({
                "file_path": file_path,
                "sharpness": sharpness,
                "transform_matrix": transform_matrix.tolist()
            })
    return frames


def csv_add(tracker_streams, camera2robot_file, transform_file):
    frames = process_frames(tracker_streams, camera2robot_file)
    with open(transform_file, "r+") as json_file:
        data = json.load(json_file)
        data['frames'] = frames
        json_file.seek(0)
        json.dump(data, json_file, indent=4)
        json_file.truncate()


streams = stream_read(args.pose_file)
csv_add(streams, args.transforms_matrix, args.camera_file)
