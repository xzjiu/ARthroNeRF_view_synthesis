import cv2
from sksurgerynditracker.nditracker import NDITracker
import keyboard  # using module keyboard
import time
import os
import csv
import numpy as np
import pickle
import argparse
import ast

def parse_args():
    parser = argparse.ArgumentParser(description="Run instant neural graphics primitives with additional configuration & output options")
    parser.add_argument("--gt", type=str, default="")
    parser.add_argument("--userid", type=str,default="0")
    parser.add_argument("--test_file", type=str,default="")
    return parser.parse_args()


"""
This file is used both for collect ground truth data and also for evaluate during test.
"""

def stream_read(file):
    stream_data = []
    with open(file, 'r') as csvfile:
        data = csv.reader(csvfile)
        for row in data:
            stream_data.append(row)
    return stream_data


def get_plate_pos():
    """
    Return plate base position
    """
    # TODO: ADD PLATE ROM
    PLATE_ROM = "C:/Users/camp/Documents/Xinrui Zou/Plate-ROM/plate2.rom"

    SETTINGS = {
        "tracker type": "polaris",
        "romfiles" : [PLATE_ROM]
            }
    TRACKER = NDITracker(SETTINGS)

    TRACKER.start_tracking()
    total_frame = 10
    base_pose = []
    for i in range(total_frame):
        port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
        print("TRACK")
        print("port handles: ", port_handles)
        print("timestamps: ", timestamps)
        print("framenumbers: ", framenumbers)
        print("quality: ", quality)
        for t in tracking:
            print(t)
        base_pose.append(tracking[0])

    TRACKER.stop_tracking()
    TRACKER.close()

    with open('base_pose.pickle', 'wb') as handle:

        pickle.dump(base_pose[0], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return base_pose[0]

def calculate_relative_positions(base_pos, stream):
    relative_stream = []
    for i, data in stream:
        # Assuming the last column of both base_pos and data arrays hold the relevant 3D positions
        base_column = base_pos[:3, -1]
        data_column = data[:3, -1]
        
        # Subtract the two columns to get the relative position
        relative_data = data_column - base_column
        relative_stream.append([i, list(relative_data)])
    return relative_stream
    
def get_tool_pose(base_pos, out_put_file):
    """
    if out_put_file = None, then the function will return the ralative position of tool based on the plate
    if out_put_file != None, the funtion will save the result to the input path
    """

    TOOL_ROM = "C:/Users/camp/Documents/Xinrui Zou/tracking_array_v3.rom"

    SETTINGS = {
        "tracker type": "polaris",
        "romfiles" : [TOOL_ROM]
            }
    TRACKER = NDITracker(SETTINGS)

    TRACKER.start_tracking()

    
    stream_list = []
    i = 0
    key_held = False
    last_pressed_time = 0
    while True:
        port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()

        # for t in tracking:
        #     print(t)

        if keyboard.is_pressed('q'):
            print("quit!!")
            break
        
        # # TODO: justify whether it is None if you want
        # elif keyboard.is_pressed('c'):
        #     tool2camera = np.array([
        #         [1.79233051e-01, 7.81816542e-02, 9.80695234e-01, 2.69621530e+02],
        #         [2.05724419e-01, -9.77777731e-01, 4.03506246e-02, -2.86812080e+00],
        #         [9.62056639e-01, 1.94520791e-01, -1.91333961e-01, -6.97097727e+01],
        #         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        #     ])
        #     transformed_data = tracking[0] @ tool2camera
        #     stream.append([i, transformed_data])
        #     print('stream', stream)
            
        #     relative_stream = calculate_relative_positions(np.array(base_pos), stream)  # Converted base_pos to a NumPy array
        #     print(f"Relative stream: {relative_stream}")
        #     stream_list = []
        #     stream_list += relative_stream
        #     print('Updated stream_list:', stream_list)  # Print the updated stream_list
        #     keyboard.wait('c')
        #     i = i+1

        if keyboard.is_pressed('k'):
            if keyboard.is_pressed('k') and (time.time() - last_pressed_time >= 0.5):
                if not key_held:
                    x = np.array([
                    [1.79233051e-01, 7.81816542e-02, 9.80695234e-01, 2.69621530e+02],
                    [2.05724419e-01, -9.77777731e-01, 4.03506246e-02, -2.86812080e+00],
                    [9.62056639e-01, 1.94520791e-01, -1.91333961e-01, -6.97097727e+01],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
                    ])
                    transformed_data = tracking[0] @ x
                    stream = [[i, transformed_data]]
                    print('stream', stream)
                    
                    relative_stream = calculate_relative_positions(np.array(base_pos), stream)  # Converted base_pos to a NumPy array
                    print(f"Relative stream: {relative_stream}")
                    stream_list.append(relative_stream)
                    print('Updated stream_list:', stream_list)  # Print the updated stream_list
                    i = i+1
                    key_held = True
                    last_pressed_time = time.time()
                else:
                    key_held = False
        
        

            
    TRACKER.stop_tracking()
    TRACKER.close()

    

    print('stream_list', stream_list)
    if out_put_file != None:
        with open(out_put_file, 'w') as file:
            writer = csv.writer(file)
            writer.writerows(stream_list)

    # TODO: Process related position: position between plate pos and tool pose
    # process stream between the tracking matrix and the plate position


    return stream_list




    # Implement your logic here for calculating the relative positions
    # This is just a placeholder implementation.
    
    

if __name__ == "__main__":
    # GT = True means to store the ground truth file
    args = parse_args()
    if args.gt != "":
        GT = True
        ground_truth = args.gt
    else:
        GT = False
    
    out_put_file = "collect_data_user_" + args.userid + ".csv"

    if GT:
        # TODO: compare the result of two points
        ground_truth_stream = stream_read(ground_truth)
        print(ground_truth_stream)
        if args.test_file != "":
            tool_stream = stream_read(args.test_file)
        else:
            base_pos = get_plate_pos()
            print("Base Pos:", base_pos)
            tool_stream = get_tool_pose(base_pos, None)
        # Compare ground_truth_stream and tool_stream ...
        

    else:
        # Store data captured by tool
        base_pos = get_plate_pos()
        print("Base Pos:", base_pos)
        get_tool_pose(base_pos, out_put_file)
    
