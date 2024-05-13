import cv2
from sksurgerynditracker.nditracker import NDITracker
import time
import os
import csv
import argparse

parser = argparse.ArgumentParser(description='Collect images and tracking information')
parser.add_argument('--row_path', type=str, default="C:/Users/camp/Documents/Xinrui Zou/tracking_array_v3.rom",
                    help='tracking pattern description')
parser.add_argument('--img_path', type=str, default='./images_carbox/',
                    help='output image path')
parser.add_argument('--pose_file', type=str, default="pivot_carbox.csv",
                    help='robot poses with corresponding image path')
args = parser.parse_args()


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def main(row_path, image_path):
    """
    Collect images and tracking information
    row_path: Tracker definition
    image_path: directory that store the images
    out_put_file: image-tracking pair in csv format
    """
    # DEFINE TRACKER
    SHOW = True
    ROM_PATH = row_path
    SETTINGS = {
        "tracker type": "polaris",
        "romfiles": [ROM_PATH]
    }
    TRACKER = NDITracker(SETTINGS)

    # open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open the camera.")
        return
    mkdir(image_path)
    TRACKER.start_tracking()
    i = 0
    stream_list = [['ID', 'Image', 'Timestamps', 'Tracking']]
    while True:
        ret, frame = cap.read()
        port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()

        if not ret:
            print("Error: Unable to read the frame.")
            break

        if SHOW:
            cv2.imshow('Camera Output', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            print(tracking[0])
            image_name = image_path + str(i) + ".png"
            cv2.imwrite(image_name, frame)  # Save image
            i += 1
            stream_list.append([i, image_name, timestamps, tracking[0]])
            print()
            time.sleep(0.2)

    TRACKER.stop_tracking()
    TRACKER.close()
    cap.release()
    cv2.destroyAllWindows()

    return stream_list


if __name__ == "__main__":
    stream_list = main(args.row_path, args.img_path)
    # Write and save image information
    with open(args.pose_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(stream_list)
