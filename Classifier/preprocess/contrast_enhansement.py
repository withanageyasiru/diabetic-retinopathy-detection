import cv2
from tqdm import tqdm


def contrast_enhancement(images):
    '''

    :param images:
    :return:
    creating a Histograms Equalization
    of a image using cv2.equalizeHist()
    '''
    print("Contrast Enhancement Processing...")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i in tqdm(range(images.shape[0])):
        images[i, :, :, 0] = clahe.apply(images[i, :, :, 0])
        images[i, :, :, 1] = clahe.apply(images[i, :, :, 1])
        images[i, :, :, 2] = clahe.apply(images[i, :, :, 2])
    print("Contrast Enhancement Process Completed")
    return images
