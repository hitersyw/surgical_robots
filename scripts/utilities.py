"""
Have this be a separate script of utility methods.
Import it inside the detect_images.py code.
"""

#import rospy
#from sensor_msgs.msg import Image
import cv2
import numpy as np
from sklearn.feature_extraction import image
from keras.models import load_model
from keras.utils import np_utils


def keras_test():
    """ Some Keras testing here. Be sure I am consistent in data loading and
    preprocessing! TODO later put the model loading inside the subscriber,
    though I can help re-shape as needed.
    """
    X_test = np.load('final_data/X_test.npy')
    y_test = np.load('final_data/y_test.npy')

    te0, te1, te2 = X_test.shape
    X_test = X_test.reshape(te0, te1, te2, 1)
    X_test /= 255 # don't forget!
    n_classes = 2
    Y_test = np_utils.to_categorical(y_test, n_classes)

    model = load_model("networks/cnn_cifar_keras_style.h5")
    #score = model.evaluate(X_test, Y_test, verbose=1)
    #print("Test score: {}".format(score[0]))
    #print("Test accuracy: {}".format(score[1]))
    predictions = model.predict(X_test, verbose=1)
    print(predictions.shape)
    print(np.argmax(predictions, axis=1))


def rgb2gray(rgb):
    """ The usual, from StackOverflow. """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def generate_patches(im, raw_size, scaled_size, stride):
    """ Generates patches and centroids from an input image.
    
    Args:
        im: A **grayscale** image from the robot's camera, which I assume has
            shape (1080,1920).
        raw_size: A 2-D tuple representing the raw (pixel) sizes of each patch.
        scaled_size: A 2-D tuple representing the **resized** version of these
            patches, using the same linear interpolation from training.
        stride: The amount we skip when extracting new patches.
        
    Returns:
        A tuple consisting of: (1) a 3-D array of patches of size (N,d1,d2)
        where (d1,d2)=scaled_size, and (2) an array of centroids of size (N,2)
        where the second axis represents the centroid, with coordinates rounded
        to the nearest integer.
    """
    patches = []
    centroids = []
    x,y = 0,0
    dx,dy = raw_size
    maxX,maxY = im.shape
  
    for x in range(0, maxX-dx, stride):
        for y in range(0, maxY-dy, stride):
            patch = cv2.resize(im[x:x+dx, y:y+dy],
                               scaled_size,
                               interpolation=cv2.INTER_LINEAR)
            patches.append(patch)
            cx = x + dx/2
            cy = y + dy/2
            centroids.append(np.array([cx,cy]))

    return np.array(patches), np.array(centroids)


if __name__ == "__main__":
    # keras test, it works
    #keras_test()

    # Some test cases here with patches.
    im = rgb2gray(np.load("np_image/left0.npy"))
    print("Loaded image with shape {}.".format(im.shape))
    patches, centroids = generate_patches(im, 
                                          raw_size=(100,100), 
                                          scaled_size=(32,32), 
                                          stride=100)
    print(patches.shape)
    print(centroids.shape)
    print(centroids)
