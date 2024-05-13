import cv2
from sksurgerynditracker.nditracker import NDITracker
import time
import os
import csv

interval = 2

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def main(row_path, image_path, out_put_file):
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
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    if not cap.isOpened():
        print("Error: Unable to open the camera.")
        return
    if not os.path.exists(image_path):
        mkdir(image_path)
    dirListing = os.listdir(image_path)
    print(dirListing)
    i = len(dirListing)*interval + 1
    print(i)
    TRACKER.start_tracking()
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
            i += 1
            if i%interval == 0:
                image_name = image_path + str(i) + ".png"
                cv2.imwrite(image_name, frame)  # Save image
           
                stream_list.append([i, image_name, timestamps,tracking[0]])

    TRACKER.stop_tracking()
    TRACKER.close()
    cap.release()
    cv2.destroyAllWindows()

    print(stream_list)

    # Write and save image information
    if os.path.exists(out_put_file):
        stream_list = stream_list[1:]
    with open(out_put_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(stream_list)


if __name__ == "__main__":
    main("C:/Users/camp/Documents/Xinrui Zou/tracking_array_v3.rom", './images/', "./pivot.csv")