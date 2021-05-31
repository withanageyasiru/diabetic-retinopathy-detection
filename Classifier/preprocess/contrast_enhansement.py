import cv2


def contrast_enhancement(images):
    '''

    :param images:
    :return:
    creating a Histograms Equalization
    of a image using cv2.equalizeHist()
    '''

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i in range(images.shape[0]):
        images[i, :, :, 0] = clahe.apply(images[i, :, :, 0])
        images[i, :, :, 1] = clahe.apply(images[i, :, :, 1])
        images[i, :, :, 2] = clahe.apply(images[i, :, :, 2])

    return images
