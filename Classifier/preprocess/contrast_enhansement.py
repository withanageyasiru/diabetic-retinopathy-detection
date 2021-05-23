
import cv2


def contrast_enhancement(image):
    # creating a Histograms Equalization
    # of a image using cv2.equalizeHist()
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    clahe_b = clahe.apply(b)
    clahe_g = clahe.apply(g)
    clahe_r = clahe.apply(r)

    return cv2.merge((clahe_b, clahe_g, clahe_r))

