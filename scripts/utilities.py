"""
Have this be a separate script of utility methods.
Import it inside the detect_images.py code.
"""

#import rospy
#from sensor_msgs.msg import Image
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


def generate_patches(im, size, stride):
    """ Generates patches. For now I'm just using the scikit-learn method 
    but we really should have our own implementation. It's not hard, just 
    a bunch of annoying indexing. Size should be 2-D since the third channel
    should not be split up.
    """
    patches = []
    x,y,z = 0,0,0
    dx,dy = size
    maxX,maxY,_ = im.shape
  
    for x in range(0, maxX-dx, stride):
        for y in range(0, maxY-dy, stride):
          patches.append(im[x:x+dx, y:y+dy, :]) 
    return np.array(patches)


if __name__ == "__main__":
    # keras test, it works
    #keras_test()

    # Some test cases here with patches.
    im = np.load("np_image/left0.npy")
    print("Loaded image with shape {}.".format(im.shape))
    patches = generate_patches(im, size=(400,400), stride=200)
    print(patches.shape)
