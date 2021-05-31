import numpy as np


def intensity_component(images):
    """

    :param images:
    :return:
    # A 12-value tuple which is a transform matrix for dropping
    # blue channel (in this case)
    # matrix = (0.4, 0, 0, 0.2,
    #           0, 0.8, 0, 0.1,
    #           0, 0, 0.1, 0.1)
    # return image.convert("RGB", matrix)
    """

    images[:, :, :, 0] = np.zeros([images.shape[1], images.shape[2]])
    images[:, :, :, 1] = images[:, :, :, 1] * 0.966
    images[:, :, :, 2] = images[:, :, :, 2] * 0.421
    images = np.uint8(images)

    return images
