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

    filesize = os.stat(filename).st_size
    logging.info(f"file size {filesize} bytes")
    original_image = Image.open(filename)
    original_array = np.array(original_image)
    dimensions = original_image.size
    logging.info(f"x: {dimensions[0]} y: {dimensions[1]}");

    # '''create gabor filter'''
    # gfilters = create_gaborfilter()

    # '''get filtered image array'''
    # gf_array = apply_filter(original_array, gfilters)

    logging.info("applying gaussian blur")
    gaussian_blurred = cv2.GaussianBlur(original_array, (5, 5), 200)

    logging.info("detecting edges")
    canny = cv2.Canny(gaussian_blurred, 1, 200)

    '''make pil image from array'''
    gf_image = Image.fromarray(canny)

    '''display image'''
    plt.imshow(np.array(gf_image))
    plt.tight_layout
    plt.show()

if __name__ == "__main__":
    main()
