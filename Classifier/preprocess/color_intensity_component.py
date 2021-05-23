from PIL import Image


def intensity_component(image):
    # Creating a image object, of the sample image
    img = Image.open(image)

    # A 12-value tuple which is a transform matrix for dropping
    # blue channel (in this case)
    matrix = (0.4, 0, 0, 0.2,
              0, 0.8, 0, 0.1,
              0, 0, 0.1, 0.1)
    return img.convert("RGB", matrix)

# img = Image.open('10_left.jpeg')
#
# # A 12-value tuple which is a transform matrix for dropping
# # green channel (in this case)
# matrix = (0.4, 0, 0, 0.2,
#           0, 0.8, 0, 0.1,
#           0, 0, 0.1, 0.1)
#
# # Transforming the image to RGB using the aforementioned matrix
# img = img.convert("RGB", matrix)
#
# # Displaying the image
# img.show()
