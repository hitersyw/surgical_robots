"""
Have this be a separate script of utility methods.
Import it inside the detect_images.py code.
"""

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
    score = model.evaluate(X_test, Y_test, verbose=1)
    print("Test score: {}".format(score[0]))
    print("Test accuracy: {}".format(score[1]))
    predictions = model.predict(X_test, verbose=1)
    print(predictions.shape)
    print(np.argmax(predictions, axis=1))


def rgb2gray(rgb):
    """ The usual, from StackOverflow. """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def get_patches(im, raw_size, scaled_size, stride, save=False):
    """ Generates patches and centroids from an input image. It also includes 
    functionality to save the original-sized patches for manual inspection. Note
    also that we do NOT scale (divide by 255) here, nor do we zero-center. That
    comes outside of this code, and means I we directly look at patches_original
    to inspect images.
    
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
    patches_original = []
    x,y = 0,0
    dx,dy = raw_size
    maxX,maxY = im.shape
  
    for x in range(0, maxX-dx, stride):
        for y in range(0, maxY-dy, stride):
            if save:
                patches_original.append(im[x:x+dx, y:y+dy])
            patch = cv2.resize(im[x:x+dx, y:y+dy],
                               scaled_size,
                               interpolation=cv2.INTER_LINEAR)
            patches.append(patch)
            cx = x + dx/2
            cy = y + dy/2
            centroids.append(np.array([cx,cy]))

    if save:
        np.save("original_patches", np.array(patches_original))
    return np.array(patches), np.array(centroids)


def test_davinci_patches():
    """ For testing patches seen by davinci and inspect performance. Don't call
    this while running davinci itself; do it after the experiment. Make sure the
    inspection is done on the non-centered, non-scaled, and non-resized data.
    """
    outfile = "misc/"
    patches = np.load("misc/patches_davinci.npy")
    print("Loaded patches of shape {}".format(patches.shape))
    for i in range(patches.shape[0]):
        cv2.imwrite(outfile+ "patch_" +str.zfill(str(i),3)+ ".jpg", patches[i])


if __name__ == "__main__":
    # Keras test, it should work
    #keras_test()

    # Some test cases here with patches.
    #im = rgb2gray(np.load("np_image/left0.npy"))
    #print("Loaded image with shape {}.".format(im.shape))
    #patches, centroids = get_patches(im, 
    #                                 raw_size=(100,100), 
    #                                 scaled_size=(32,32), 
    #                                 stride=100)

    # Test the patches found from davinci (requires file outside of github).
    test_davinci_patches()
    pass
