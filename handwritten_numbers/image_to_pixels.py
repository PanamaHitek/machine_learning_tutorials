import os
import numpy as np
from PIL import Image
import pandas as pd

csv_array = []
index = 0

for pictures in os.scandir("images/resized_digits"):
    pixel_array = []
    original_image = Image.open(pictures.path)
    pixels = original_image.load()
    index = index + 1
    for i in range(28):
        for j in range(28):
            p = int(pixels[j, i])
            if (p > 50):
                pixel_array.append(p)
            else:
                pixel_array.append(0)
    csv_array.append(pixel_array)

csv_array = np.asarray(csv_array)
pd.DataFrame(csv_array).to_csv("../datasets/custom/custom_mnist.csv", header=None)
