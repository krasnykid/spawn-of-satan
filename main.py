import numpy as np
import cv2
import logging
import matplotlib.pyplot as plt
from PIL import Image

# Basic configuration
logging.basicConfig(
    level=logging.INFO,  # Set logging.info level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s [%(funcName)s] %(levelname)s - %(message)s'  # Log message format
)

def find_contours(src_array, src_width, src_height):
    ratio = src_height / 500

    logging.info(f"ratio: {ratio}")

    height = int(src_height / ratio)
    width = int(src_width / ratio)

    logging.info("resizing image")
    resized_image = cv2.resize(src_array, (width, height))

    logging.info("converting to grayscale")
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY, hint = 4)

    logging.info("gaussian bloor")
    gblurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    logging.info("canny")
    canned_image = cv2.Canny(gblurred_image, 75, 200)

    logging.info("detecting contour")
    contours_found = cv2.findContours(canned_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_found[0] if len(contours_found) == 2 else contours_found[1]

    # Sort contours by area, descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return contours


def sortPoints(src_points):
    result = [None] * 4

    sum_comparator = lambda p: p[0] + p[1]
    diff_comparator = lambda p: p[1] - p[0]

    result[0] = min(src_points, key=sum_comparator)

    # bottom-right corner = maximal sum
    result[2] = max(src_points, key=sum_comparator)

    # top-right corner = minimal difference
    result[1] = min(src_points, key=diff_comparator)

    # bottom-left corner = maximal difference
    result[3] = max(src_points, key=diff_comparator)

    return np.array(result, dtype=np.float32)

def getQuad(contours, src_width, src_height):
    ratio = src_height / 500  # for normalization if needed later

    for c in contours:
        # Convert contour to float32
        c2f = c.astype(np.float32)

        # Approximate the contour
        peri = cv2.arcLength(c2f, True)
        approx = cv2.approxPolyDP(c2f, 0.02 * peri, True)

        # Reshape to (4, 2) and check for quadrilateral
        if len(approx) == 4:
            points = approx.reshape(4, 2)
            sorted_points = sortPoints(points.astype(np.float32))
            return tuple(sorted_points)

    return None  # If no quadrilateral was found

def main():
    filename = "photos/git2.jpg"

    src = Image.open(filename)
    src_array = np.array(src)
    logging.info(f"x: {src.width} y: {src.height}");

    contours = find_contours(src_array, src.width, src.height)

    if not contours:
        logging.error("no contours found")
        return

    quad = getQuad(contours, src.width, src.height)

    if not quad:
        logging.error("no quad")
        return

    # Compute width and height of the new image
    (tl, tr, br, bl) = quad

    widthA = float(np.linalg.norm(br - bl))
    widthB = float(np.linalg.norm(tr - tl))
    maxWidth = int(max(widthA, widthB))

    heightA = float(np.linalg.norm(tr - br))
    heightB = float(np.linalg.norm(tl - bl))
    maxHeight = int(max(heightA, heightB))

    quad = np.array(quad, dtype=np.float32)

    # Define destination points for "bird's eye view"
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    # Compute perspective transform matrix and warp
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(src_array, M, (maxWidth, maxHeight))

    '''make PIL image from array'''
    final = Image.fromarray(warped)

    '''display image'''
    plt.imshow(np.array(final))
    plt.tight_layout
    plt.show()

if __name__ == "__main__":
    main()
