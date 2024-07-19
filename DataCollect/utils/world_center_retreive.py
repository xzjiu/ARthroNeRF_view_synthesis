from sksurgerynditracker.nditracker import NDITracker
import pickle
import numpy as np

# Change Rom Path based on your marker array
ROM_PATH = "C:/Users/camp/Documents/Xinrui Zou/tracking_array_v3.rom"

SETTINGS = {
    "tracker type": "polaris",
    "romfiles" : [ROM_PATH]
        }
TRACKER = NDITracker(SETTINGS)

TRACKER.start_tracking()
total_frame = 10
start_pose = []
for i in range(total_frame):
    port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
    print("TRACK")
    print("port handles: ", port_handles)
    print("timestamps: ", timestamps)
    print("framenumbers: ", framenumbers)
    print("quality: ", quality)
    for t in tracking:
        print(t)
    start_pose.append(tracking[0])


TRACKER.stop_tracking()
TRACKER.close()

start_pose = np.asarray(start_pose)

avg_translation = np.mean(start_pose[:, :3, 3], axis=0)

# construct a new transformation matrix for the world center
world_center = np.identity(4)
world_center[:3, 3] = avg_translation

with open('world_center.pkl', 'wb') as f:
        pickle.dump(start_pose[0], f)