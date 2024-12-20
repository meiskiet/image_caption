import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from textwrap import wrap

def read_image(image_path, img_size=224):
    """
    Reads an image from the given path, resizes it, and converts it to a normalized array.
    """
    img = load_img(image_path, target_size=(img_size, img_size), color_mode="rgb")
    img = img_to_array(img)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

def display_images_with_captions(dataframe, image_dir, num_images=15):
    """
    Displays a grid of images with captions.
    Args:
        dataframe: DataFrame containing 'image' and 'caption' columns.
        image_dir: Directory where the images are stored.
        num_images: Number of images to display.
    """
    dataframe = dataframe.sample(num_images).reset_index(drop=True)
    plt.figure(figsize=(20, 20))
    n = 0

    for i in range(num_images):
        n += 1
        plt.subplot(5, 5, n)
        plt.subplots_adjust(hspace=0.7, wspace=0.3)
        img_path = f"{image_dir}/{dataframe.image[i]}"
        image = read_image(img_path)
        plt.imshow(image)
        plt.title("\n".join(wrap(dataframe.caption[i], 20)))
        plt.axis("off")
    plt.show()
