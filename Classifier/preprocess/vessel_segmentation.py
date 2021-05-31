import cv2
import numpy as np


def vessel_segmentation(images):
    """

    :param images:
    :return:
    """
    image_set = images.copy()
    for i, img in enumerate(image_set):
        # convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # apply morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)

        # divide gray by morphology image
        division = cv2.divide(gray, morph, scale=255)

        # threshold
        thresh = cv2.threshold(division, 0, 255, cv2.THRESH_OTSU)[1]

        # invert
        thresh = 255 - thresh

        # find contours and discard contours with small areas
        mask = np.zeros_like(thresh)
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        area_thresh = 10000
        for cntr in contours:
            area = cv2.contourArea(cntr)
            if area > area_thresh:
                cv2.drawContours(mask, [cntr], -1, 255, 2)

        # apply mask to thresh
        # result1 = cv2.bitwise_and(thresh, mask)
        mask = cv2.merge([mask, mask, mask])
        result2 = cv2.bitwise_and(img, mask)

        image_set[i] = result2

    return image_set
