import math
from sksurgerynditracker.nditracker import NDITracker
import numpy as np
import time
romfile_path_brain = "c:/Users/camp/Documents/Xinrui Zou/Tool-ROM/tool_ar.rom"
romfile_path_tool = "c:/Users/camp/Documents/Xinrui Zou/Tool-ROM/tool_new.rom"
SETTINGS = {"tracker type": "polaris", "romfiles": [romfile_path_brain, romfile_path_tool]}
TRACKER = NDITracker(SETTINGS)
TRACKER.start_tracking()

try:
    while True:
        # Fetch the current frame of tracking data.
        port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()

        # Access tracking data for both items.
        frame1 = tracking[0] if len(tracking) > 0 else None
        frame2 = tracking[1] if len(tracking) > 1 else None

        # Print the tracking data if available.
        if frame1 is not None:
            print("Frame 1 Tracking:", frame1)
        if frame2 is not None:
            print("Frame 2 Tracking:", frame2)
        
        # Calculate the inverse of Frame 1's matrix
        frame1_inv = np.linalg.inv(frame1)

        # Calculate the transformation of Frame 2 relative to Frame 1
        frame2_relative_to_frame1 = np.dot(frame1_inv, frame2)

        print("Frame 2 position and orientation relative to Frame 1:\n", frame2_relative_to_frame1)

        # Wait a bit before fetching the next frame to not overwhelm the output.
        time.sleep(1)  # Sleep for 1 second. Adjust this value as needed.

except KeyboardInterrupt:
    print("Tracking stopped.")
    TRACKER.stop_tracking()

