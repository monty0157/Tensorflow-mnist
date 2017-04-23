from PIL import Image
import numpy as np
import tensorflow as tf

def image_import(image):
    #Load image from data
    image_open = Image.open(image)
    image_convert = image_open.convert('RGB')
    image_array = np.asarray(image_convert, dtype="uint8")

    return image_array

def downsample(image_array, size):

    #downsample image to given size
    image_downsample = image_array[::image_array.shape[0]/size,::image_array.shape[1]/size]

    #get image shape and finish downsampling
    image_get_shape = image_downsample.shape
    image_downsample = image_downsample[image_get_shape[0]-size:, image_get_shape[1]-size:]

    '''with tf.Session() as sess:
        image_downsample = tf.image.resize_images(image_array, (size,size)).eval()'''
    return image_downsample

def rgb_to_gray(data):
    grey = np.zeros((data.shape[0], data.shape[1]))

    for row in range (len(data)):
        for column in range(len(data[row])):
            #take average pixel value across channels and reduce range to between 0 and 1
            grey[row][column] = np.average(data[row][column])

    return grey

#Webcam mirrors image
def horizontal_mirror(image_array):
    mirror = np.fliplr(image_array)

    return mirror

def image_round(image_array):
    image_array = ((255-image_array)/255)
    #Round values of pixels
    for pixel in range (len(image_array)):
        image_array[pixel] = np.around(image_array[pixel])
    return image_array

def image_processed(image, size, mirrored_image=True):
    image_array = image_import(image)
    image_downsample = downsample(image_array, size)
    convert_channels = rgb_to_gray(image_downsample)
    _round = image_round(convert_channels)
    mirror = horizontal_mirror(_round)

    if(mirrored_image == False):
        return _round

    return mirror
