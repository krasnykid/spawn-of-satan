import numpy as np
import os
import cv2
import logging
import matplotlib.pyplot as plt
from PIL import Image
from gfilter import create_gaborfilter, apply_filter

# Basic configuration
logging.basicConfig(
    level=logging.INFO,  # Set logging.info level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s [%(funcName)s] %(levelname)s - %(message)s'  # Log message format
)

def main():
    filename = "photos/PXL_20250521_090348880.jpg"

    gray_image = None
    canned_image = None
    resized_image = None

    src = Image.open(filename)
    src_array = np.array(src)
    logging.info(f"x: {src.width} y: {src.height}");

    ratio = src.height / 500

    logging.info(f"ratio: {ratio}")

    height = int(src.height / ratio)
    width = int(src.width / ratio)
    pixels = width * height

    logging.info("resizing image")
    resized_image = cv2.resize(src_array, (width, height))

    logging.info("converting to grayscale")
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY, hint = 4)

    logging.info("gaussian bloor")
    gblurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    logging.info("canny")
    canned_image = cv2.Canny(gblurred_image, 75, 200)

    # '''create gabor filter'''
    # gfilters = create_gaborfilter()

    # '''get filtered image array'''
    # gf_array = apply_filter(original_array, gfilters)

    # logging.info("applying gaussian blur")
    # gaussian_blurred = cv2.GaussianBlur(src_array, (5, 5), 200)

    # logging.info("detecting edges")
    # canny = cv2.Canny(gaussian_blurred, 1, 200)

    '''make pil image from array'''
    final = Image.fromarray(canned_image)

    '''display image'''
    plt.imshow(np.array(final))
    plt.tight_layout
    plt.show()

if __name__ == "__main__":
    main()
