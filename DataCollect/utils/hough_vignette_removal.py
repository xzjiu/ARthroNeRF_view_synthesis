import cv2
import numpy as np
import argparse


def convert_im_to_rgba(rgb, mask):
    """
    Converts an RGB image to RGBA, using the given mask as the alpha channel.
    """
    rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = mask
    return rgba

def detect_centered_circle(imPth):
    """
    reads image, detects vignette using hough transform and finds the closest to the center
    """
    img = cv2.imread(imPth)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 15)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=rows/8,
                               param1=100, param2=30, minRadius=int(img.shape[0]/4), maxRadius=int(img.shape[0]/2*0.85))
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        center = np.array([img.shape[1] // 2, img.shape[0] // 2])  # Calculate image center coordinates
        closest_circle = min(circles, key=lambda x: np.linalg.norm(center - x[:2]))
        return closest_circle
    else:
        return None

def convert_circle_to_mask(img, circle):
    mask = np.zeros(img.shape[:2], np.uint8)
    dilate = -5
    cv2.circle(mask, (circle[0], circle[1]), circle[2]+dilate, 255, -1)
    return mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imPth', type=str, default='/home/bigss/Documents/instant-ngp/data/nerf/stryker/gum_box/images/0001.jpg')
    parser.add_argument('--amtImages', type=int, default=122)
    imPthSkeleton = parser.parse_args().imPth
    amtImages = parser.parse_args().amtImages
    print(f"Looping trough {amtImages} images...")
    for i in range(1, amtImages):
        imPth = imPthSkeleton.replace("0001", str(i).zfill(4))
        print(imPth)
        center_circle = detect_centered_circle(imPth)
        rgb = cv2.imread(imPth)       
        mask=convert_circle_to_mask(rgb, center_circle)
        rgba = convert_im_to_rgba(rgb, mask) 
        cv2.imwrite(f'output_rgba/{str(i).zfill(4)}.png', rgba)
        if center_circle is not None:
            x, y, r = center_circle
            cv2.circle(rgb, (x, y), r, (0, 255, 0), 2)
            cv2.circle(rgb, (x, y), r-5, (255, 0, 0), 2)
        else:
            print("No circle detected")
        cv2.imwrite(f'debug_hough_circle/{str(i).zfill(4)}.png', rgb)
    print("Detected circle is stored in debug_hough_circle and output_rgba folders")


if __name__ == '__main__':
    main()
