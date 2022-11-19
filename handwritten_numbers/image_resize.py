import os
from PIL import Image, ImageOps

index = 0
for pictures in os.scandir("images/cropped_digits"):  # Scan images in this folder
    original_image = Image.open(pictures.path)  # Load the image
    rgb_image = original_image.convert("RGB")  # Convert loaded image to RGB
    grayscale_image = ImageOps.grayscale(rgb_image)  # Convert to grayscale
    grayscale_image = ImageOps.invert(grayscale_image)  # Invert colors (black to white and vice versa)
    resized_image = grayscale_image.resize((28, 28))  # Resize images to the needed 28x28 format
    resized_image.save("images/resized_digits/" + str(index) + ".jpg")  # Save transformed images to folder
    index = index + 1
