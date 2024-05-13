import cv2
import lpips
import numpy as np
from skimage import io, img_as_float32
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import os
import torch
from torchvision import transforms

# Initialize LPIPS
lpips_model = lpips.LPIPS(net='alex')  # Using AlexNet

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

def compute_metrics(image_path, reference_path):
    # Load images with OpenCV
    image_cv = cv2.imread(image_path)
    reference_cv = cv2.imread(reference_path)

    # Convert from BGR to RGB
    image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    reference = cv2.cvtColor(reference_cv, cv2.COLOR_BGR2RGB)

    # Resize the image to match the reference image's dimensions
    if image.shape != reference.shape:
        image = cv2.resize(image, (reference.shape[1], reference.shape[0]))

    # Convert images to PIL format for LPIPS (through np.array to PIL Image)
    image_pil = Image.fromarray(image)
    reference_pil = Image.fromarray(reference)

    # Convert images to tensors
    image_tensor = transform(image_pil).unsqueeze(0)  # Add batch dimension
    reference_tensor = transform(reference_pil).unsqueeze(0)  # Add batch dimension

    # Use GPU if available
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
        reference_tensor = reference_tensor.cuda()
        lpips_model.cuda()

    # Compute LPIPS
    lpips_score = lpips_model(image_tensor, reference_tensor)

    # Convert images to float32 numpy arrays for PSNR and SSIM
    image_np = np.array(image) / 255.0
    reference_np = np.array(reference) / 255.0

    # Compute PSNR
    psnr_score = psnr(reference_np, image_np, data_range=1)

    # Compute SSIM
    ssim_score = ssim(reference_np, image_np, data_range=1, channel_axis=-1)

    return lpips_score, psnr_score, ssim_score

# Paths to your folders
test_images_folder = "tracking_screenshot"
reference_images_folder = "images"

lpips_scores, psnr_scores, ssim_scores = [], [], []

# Loop through test images
for filename in os.listdir(test_images_folder):
    test_image_path = os.path.join(test_images_folder, filename)
    reference_image_path = os.path.join(reference_images_folder, filename)

    if os.path.exists(reference_image_path):
        # Compute metrics
        lpips_score, psnr_score, ssim_score = compute_metrics(test_image_path, reference_image_path)
        print(f"Image: {filename}, LPIPS: {lpips_score}, PSNR: {psnr_score}, SSIM: {ssim_score}")
        lpips_scores.append(lpips_score)
        psnr_scores.append(psnr_score)
        ssim_scores.append(ssim_score)
    else:
        print(f"Reference image for {filename} not found.")

print(f"Average LPIPS: {sum(lpips_scores)/len(lpips_scores)}, PSNR: {sum(psnr_scores)/len(psnr_scores)} , SSIM: {sum(ssim_scores)/len(ssim_scores)}")