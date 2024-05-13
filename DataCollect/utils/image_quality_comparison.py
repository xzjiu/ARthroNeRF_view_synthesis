from math import log10, sqrt
import cv2
import numpy as np
import os

def PSNR(original, compressed):
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0): # MSE is zero means no noise is present in the signal .
    # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def main():
	original = cv2.imread("black_toy/images/564.png")
	compressed = cv2.imread("black_toy/test_output/564.png", 1)
	value = PSNR(original, compressed)
	print(f"PSNR value is {value} dB")

def computer_average(path):
    images = os.listdir(path+'/test_output')
    total = []
    for image in images:
        generated = path + '/test_output/' + image
        original =  path + '/images/' + image
        generated = cv2.imread(generated)
        original = cv2.imread(original, 1)
        value = PSNR(original, generated)
        total.append(value)
    print(sum(total)/len(total))
	
if __name__ == "__main__":
    main()
    computer_average('black_toy')