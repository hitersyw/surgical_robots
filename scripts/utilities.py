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


def get_processed_patches(im, raw_size, scaled_size, stride, save=False, left=True):
    """ Generates patches and centroids from an input image. It also includes 
    functionality to save the original-sized patches for manual inspection. The 
    patches and centroids that this returns should be processed so that the 
    davinci code can _directly_ call the keras code on it. That means we zero
    mean and divide by 255 here, along with reshaping for keras.
    
    Args:
        im: A **grayscale** image from the robot's camera, which I assume has
            shape (1080,1920).
        raw_size: A 2-D tuple representing the raw (pixel) sizes of each patch.
        scaled_size: A 2-D tuple representing the **resized** version of these
            patches, using the same linear interpolation from training. AND we 
            zero-center them here (which happens before dividing by 255).
        stride: The amount we skip when extracting new patches.
        left: If true, we assume we're calling this from the left camera, which
          affects the saved file name. Otherwise, the right image.
        
    Returns:
        A tuple consisting of: (1) a 4-D array of processed patches of size 
        (N,d1,d2,1), where (d1,d2)=scaled_size, and (2) an array of centroids 
        of size (N,2) where the second axis represents the centroid, with 
        coordinates rounded to the nearest integer.
    """
    patches = []
    centroids = []
    patches_original = []
    x,y = 0,0
    dx,dy = raw_size
    maxX,maxY = im.shape
    X_mean = np.load("final_data/X_train_MEAN.npy")
  
    for x in range(0, maxX-dx, stride):
        for y in range(0, maxY-dy, stride):
            if save:
                patches_original.append(im[x:x+dx, y:y+dy])
            patch = cv2.resize(im[x:x+dx, y:y+dy],
                               scaled_size,
                               interpolation=cv2.INTER_LINEAR)
            patch -= X_mean
            patches.append(patch)
            cx = x + dx/2
            cy = y + dy/2
            centroids.append(np.array([cx,cy]))

    if save:
        prefix = "right_"
        if left:
            prefix = "left_"
        np.save("misc/" +prefix+ "full_image", im)
        np.save("misc/" +prefix+ "raw_patches", np.array(patches_original))
    patches = np.array(patches)
    sh0, sh1, sh2 = patches.shape
    patches = patches.reshape(sh0, sh1, sh2, 1)
    patches /= 255 
    return patches, np.array(centroids)


def test_davinci_patches(left=True):
    """ For testing patches seen by davinci and inspect performance. Don't call
    this while running davinci itself; do it after the experiment. Make sure the
    inspection is done on the non-centered, non-scaled, and non-resized data.
    UPDATE: we can also use this to test the assocation between the left camera
    and right camera coordinate frames.
    """

    prefix="right"
    other="left"
    if left:
        prefix="left"
        other="right"

    # Load stuff .
    patches = np.load("misc/" +prefix+ "_raw_patches.npy")
    centroids = np.load("misc/" +prefix+ "_centroids.npy")
    mapping_centroids = np.load("misc/" +prefix+ "_to_" +other+ "_centroids.npy")
    image = np.load("misc/" +prefix+ "_full_image.npy")
    image_copy = np.copy(image) # For putting mapped centroids
    print("Loaded patches of shape {}".format(patches.shape))
    cv2.imwrite("misc/" +prefix+ "_full_image.jpg", image)

    # First, let's create the patches themselves.
    for i in range(patches.shape[0]):
        cv2.imwrite("misc/patches/" +prefix+ "_patch_" +str.zfill(str(i),3)+ ".jpg", patches[i])

    # Now let's draw out the location of the centroids.
    font = cv2.FONT_HERSHEY_SIMPLEX
    rectangle_i = 2000
    for (index,pt) in enumerate(centroids):
        x,y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(image, (x,y), radius=4, color=(0,0,0), thickness=2)
        cv2.putText(image, text=str(index), org=(x,y), fontFace=font, fontScale=0.8, color=(0,0,0), thickness=2)
        if (index == rectangle_i):
            cv2.rectangle(image, (x-200,y-200), (x+200,y+200), (255,255,0), 2)
    cv2.imwrite("misc/" +prefix+ "_full_image_centroids.jpg", image)
    for (index,pt) in enumerate(mapping_centroids):
        x,y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(image_copy, (x,y), radius=4, color=(0,0,0), thickness=2)
        cv2.putText(image_copy, text=str(index), org=(x,y), fontFace=font, fontScale=0.8, color=(0,0,0), thickness=2)
        if (index == rectangle_i):
            print("inside special index, x={},  y={}, prefix={}".format(x,y,prefix))
            cv2.rectangle(image_copy, (x-200,y-200), (x+200,y+200), (255,255,0), 2)
    cv2.imwrite("misc/" +prefix+ "_full_image_centroids_to_" +other+ ".jpg", image_copy)
   


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
    test_davinci_patches(left=True)
    test_davinci_patches(left=False)
