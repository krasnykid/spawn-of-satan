import numpy as np
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
    original_image = Image.open('photos/git1.jpg')
    original_array = np.array(original_image)

    '''create gabor filter'''
    gfilters = create_gaborfilter()

    '''get filtered image array'''
    gf_array = apply_filter(original_array, gfilters)

    '''make pil image from array'''
    gf_image = Image.fromarray(gf_array)

    '''display image'''
    plt.imshow(np.array(gf_image))
    plt.tight_layout
    plt.show()

if __name__ == "__main__":
    main()
