# arthro_nerf

* Attention: activate virtual env before running this code. 
(run `./venv/Scripts/activate`)
Or run `pip install -r requirements.txt`

## Transforms.json generation 

To get transforms.json used in Instant-ngp, we need following steps:

### pre-requirements:
* attach marker array to camera
* `handeye_matrix` between camera and tracking center: run `camera_hand_eye.py`.

1. create a new directory that you want to store the current images and tracking information.
2. run `python ../utils/world_center_retreive.py` to get the world center.
3. run `python ../utils/camera_stream_ndi_tracking.py` to capture images of object and the current tracking position.
4. run `python ../utils/world_center_extrinsic_mapping.py` to add frames.

### Preparation
1. build your own rom file based on the NDI marker array
2. change rom file in `camera_stream_ndi_tracking.py` and `world_center_retreive.py`