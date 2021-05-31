import cv2 as cv


def noise_removal(image):
    '''

    :param image:
    :return:
    '''

    for i in range(image.shape[0]):
        image[i, :, :, :] = cv.medianBlur(image[i, :, :, :], 5)
        # image[i, :, :, :] = cv.adaptiveThreshold(image[i, :, :, :], 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        # cv.THRESH_BINARY, 11, 2)
    return image
