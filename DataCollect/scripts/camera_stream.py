import cv2
from sksurgerynditracker.nditracker import NDITracker
import time
import os
import csv

def main():
    """
    Collect images and tracking information
    row_path: Tracker definition
    image_path: directory that store the images
    out_put_file: image-tracking pair in csv format
    """

    # open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    if not cap.isOpened():
        print("Error: Unable to open the camera.")
        return
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to read the frame.")
            break

        cv2.imshow('Camera Output', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    TRACKER.stop_tracking()
    TRACKER.close()
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
