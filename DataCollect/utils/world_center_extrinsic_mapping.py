import numpy as np
import pickle
import csv
import cv2
import os
import re
import json
import math


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

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom

def flip_matrix(m):
    c2w = np.linalg.inv(m)
    c2w[0:3,2] *= -1 # flip the y and z axis
    c2w[0:3,1] *= -1
    c2w = c2w[[1,0,2,3],:]
    c2w[2,:] *= -1 # flip whole world upside down
    return c2w

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def process_frames(tracker_file, camera2robot_file, world_center):
    frames = []
    streams = stream_read(tracker_file)
    tool2camera = np.loadtxt(camera2robot_file, dtype=float, delimiter=',')
    camera2tool = np.linalg.inv(tool2camera)
    print("HandEye Tool2Camera: \n",tool2camera)
    if not streams:
        raise Exception("No data collected!")

    origin = np.eye(4)
    HasOrigin = False
    up = np.zeros(3)
    print("World_center: \n", world_center)

    for stream in streams:
        file_path = stream['Image']
        if not os.path.exists(file_path):
            continue
        sharpness = get_sharpness(file_path)
        # TODO: get rid of vague images
        transform = stream['Tracking']
        if re.search("nan", transform):
            if os.path.exists(file_path):
                os.remove(file_path)
        else:
            transform_matrix =  np.asarray(get_matrix(transform))
            #####################---- Get camera2world matrix ---#######################

            # ####### 1. handeye calibration
            transform_matrix = camera2tool @ np.linalg.inv(transform_matrix)

            # ####### 3. use world center as origin ############
            transform_matrix = transform_matrix @ world_center @ tool2camera

            # ####### 2. flip matrix (from opencv to opengl and from w2c to c2w)
            transform_matrix = flip_matrix(transform_matrix)

            up += transform_matrix[0:3,1]
            transform_matrix[0:3, 3] *= 0.01
            frames.append({
                "file_path": file_path,
                "sharpness": sharpness,
                "transform_matrix": transform_matrix.tolist()
            })
    ###########################################################################  
    ### change orientation
    # up = up / np.linalg.norm(up)
    # print("up vector was", up)
    # R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
    # R = np.pad(R,[0,1])
    # R[-1, -1] = 1
    nframes = len(frames)
    # for f in frames:
    #     f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis
    #     # find a central point they are all looking at
    #     print("computing center of attention...")
    #     totw = 0.0
    #   totp = np.array([0.0, 0.0, 0.0])
    # for f in frames:
    #     mf = f["transform_matrix"][0:3,:]
    #     for g in frames:
    #         mg = g["transform_matrix"][0:3,:]
    #         p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
    #         if w > 0.00001:
    #             totp += p*w
    #             totw += w
    # if totw > 0.0:
    #     totp /= totw
    # print(totp) # the cameras are looking at totp
    # for f in frames:
    #     f["transform_matrix"][0:3,3] -= np.array([0.0, 0.0, 0.0])

    avglen = 0.
    for f in frames:
        f["transform_matrix"] = np.asarray(f["transform_matrix"])
        # print(f["transform_matrix"])
        avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
    avglen /= nframes
    print("avg camera distance from origin", avglen)
    for f in frames:
        f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

    for f in frames:
        f["transform_matrix"] = f["transform_matrix"].tolist()
    # # ##### postprocess-info totp avglen
    # # ##### postprocess-R R
    # with open('postprocess-R.pkl', 'wb') as f:
    #     pickle.dump(R, f)
    # with open('postprocess-info.pkl', 'wb') as f:
    #     pickle.dump([totp, avglen], f)

    with open('postprocess-info.pkl', 'wb') as f:
        pickle.dump([avglen], f)
    return frames


def csv_add(tracker_file, handeye_matrix, transform_file, world_center):
    """
    Add extrinsic matrix to existing transforms.json
    tracker_file: image - cameraPosition mapping file
    """
    frames = process_frames(tracker_file, handeye_matrix, world_center)
    with open(transform_file, "r+") as json_file:
        data = json.load(json_file)
        data['frames'] = frames
        json_file.seek(0)
        json.dump(data, json_file, indent=4)
        json_file.truncate()

def TransToJson(matrix, distortion, imgH, imgW, aabb_scale, is_fisheye, frames, outpath='transforms.json'):
    """
    Create transforms.json with key elements
    """
    AABB_SCALE = aabb_scale
    h = imgH
    w = imgW
    fl_x = matrix[0][0]
    fl_y = matrix[1][1]
    k1 = distortion[0][0]
    k2 = distortion[0][1]
    k3 = distortion[0][4]
    k4 = distortion[0][5] if len(distortion[0]) >= 6 else 0
    p1 = distortion[0][2]
    p2 = distortion[0][3]
    cx = matrix[0][2]
    cy = matrix[1][2]
    angle_x = math.atan(w / (fl_x * 2)) * 2
    angle_y = math.atan(h / (fl_y * 2)) * 2
    fovx = angle_x * 180 / math.pi
    fovy = angle_y * 180 / math.pi

    out = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "k4": k4,
        "p1": p1,
        "p2": p2,
        "is_fisheye": is_fisheye,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "aabb_scale": AABB_SCALE,
        "frames": frames,
    }

    with open(outpath, "w") as outfile:
        json.dump(out, outfile, indent=2)


if __name__ == "__main__":
    # matrix = np.array([[471.32480599,   0.00000000e+00, 319.6930523 ],
    #         [0.00000000e+00, 473.00882149, 244.39997145],
    #         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float64)
    # distortion = np.array([[ 4.67626869e-03, 6.27888081e-02, 8.77125257e-04, 2.36776524e-04, -2.60426004e-01]])

    matrix = np.array([[467.9830497,   0.00000000e+00, 320.6389447 ],
            [0.00000000e+00, 466.87870617, 246.68071929],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float64)
    distortion = np.array([[ 0.01117932, 0.13037312, 0.00050895, 0.0023196, -0.41658001]])

    imgH = 480
    imgW = 640
    is_fisheye = False
    aabb_scale = 4
    tracker_file = './pivot.csv'
    # camera2robot_file = '../camera_calibration/handeye_matrix_new_2.csv'
    camera2robot_file = '../handeye-v2/handeye_matrix.csv'
    outpath = 'transforms.json'
    world_center_file = 'world_center.pkl'

    with open(world_center_file, 'rb') as f:
        world_center = pickle.load(f)

    frames = process_frames(tracker_file, camera2robot_file, world_center)

    TransToJson(matrix, distortion, imgH, imgW, aabb_scale, is_fisheye, frames, outpath)