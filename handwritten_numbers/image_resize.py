import os
from PIL import Image, ImageOps

index = 0
for pictures in os.scandir("images/cropped_digits"):
    original_image = Image.open(pictures.path)
    rgb_image = original_image.convert("RGB")
    grayscale_image = ImageOps.grayscale(rgb_image)
    grayscale_image = ImageOps.invert(grayscale_image)
    resized_image = grayscale_image.resize((28, 28))
    resized_image.save("images/resized_digits/" + str(index) + ".jpg")
    index = index + 1
