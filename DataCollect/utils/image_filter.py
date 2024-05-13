import os
import cv2
from PIL import Image
import numpy as np

def is_blurred(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance

path = r"C:\Users\camp\GIT\arthro_nerf\LEGO\images"

# Get a list of all files in the directory
images = os.listdir(path)

# Initialize the counter
counter = 0

for image_name in images:
    # Check if the file is an image by looking at its extension
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        file_path = os.path.join(path, image_name)
        
        image = cv2.imread(file_path)

        # Check if the image was correctly loaded
        if image is not None:
            variance = is_blurred(image)
            if variance > 140:
                # Increase the counter if the variance is bigger than 140
                counter += 1
            print(f"Image: {image_name}")
            print(f"Is blurred: {variance}")
            print("--------------------------")
        else:
            print(f"Image not loaded correctly: {image_name}")
    else:
        print(f"Not an image file: {image_name}")

# Print the final count
print(f"Number of images with variance greater than 140: {counter}")
