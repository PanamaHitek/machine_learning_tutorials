import os
import numpy as np
from PIL import Image
import pandas as pd

csv_array = []  # Array to save all pixels (global array)
index = 0

for pictures in os.scandir("images/resized_digits"):  # Read al images in this path
    pixel_array = []  # Array to save all pixels from image in a row
    original_image = Image.open(pictures.path)  # Open the image
    pixels = original_image.load()  # Load pixels from image in a 2D array
    index = index + 1
    for i in range(28):
        for j in range(28):
            p = int(pixels[j, i])  # Load each pixel in the 2D array
            if (p > 50):  # Pixel cleaning condition
                pixel_array.append(p)  # Append each pixel to the row array
            else:
                pixel_array.append(0)
    csv_array.append(pixel_array)  # Append each row array to the global array

csv_array = np.asarray(csv_array)  # Convert array to numpy array
pd.DataFrame(csv_array).to_csv("../../../../datasets/custom/custom_mnist.csv", header=None)  # Save pixels in a file
