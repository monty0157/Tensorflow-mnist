from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import math

'''img = Image.open('./digit_test_2.png')
#img.show()
img = img.convert('RGB')
image = np.asarray(img, dtype="uint8")
#prevent matplotlib from converting image to negative
plt.imshow(image)
plt.show()
print(image.shape)
#image = image[::image.shape[0]/28,::image.shape[1]/28]

image = image[::math.floor(image.shape[0]/28),::math.floor(image.shape[1]/28)]

image = image[1:-1, 1:-1]

image = image.reshape(48,49)

print(image.shape)
#image_show = Image.fromarray(image)
#image_show.show()
plt.imshow(image)
plt.show()'''

'''def downsample(data):
    image = Image.open(data)
    image = image.convert('RGB')
    image = np.asarray(image, dtype="uint8")
    print(image.shape)
    image = image[::image.shape[0]/28,::image.shape[1]/28]

    return image'''
