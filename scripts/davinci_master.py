"""
This is the master code and should be executed for our experiments.

The subscribers can save very quickly, but we'll make them sleep long enough.

Also, we need to import our neural networks code here.
"""

import rospy
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker
import cv2
import cv_bridge
import numpy as np
import rospy
import scipy.misc
import pickle
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
import sys
import os

# Some stuff needed for my additions. I don't think we'll need theano GPU but
# here's how to use it if I need it.
# import theano
# import theano.config.device = 'gpu'
# import theano.config.floatX = 'float32'
from keras.models import load_model
import utilities
np.set_printoptions(suppress=True)


class DavinciMaster:

    def __init__(self):
        self.right_image = None
        self.left_image = None
        self.rcounter = 0
        self.lcounter = 0
        self.info = {'l': None, 'r': None}
        self.bridge = cv_bridge.CvBridge()

        # Load network and other information.
        self.model = load_model("networks/cnn_cifar_keras_style.h5")
        self.left_to_right = np.load("calibration_stuff/left_to_right_3x2.npy")
        self.right_to_left = np.load("calibration_stuff/right_to_left_3x2.npy")

        #========SUBSCRIBERS========#
        # image subscribers
        rospy.init_node('image_saver')
        sb = rospy.Subscriber("/endoscope/left/image_rect_color", Image, self.left_image_callback, queue_size=1)
        rospy.spin()
        # Ignore these other three subscribers for now
        #rospy.Subscriber("/endoscope/right/image_rect_color", Image, self.right_image_callback, queue_size=1)
        # info subscribers
        #rospy.Subscriber("/endoscope/left/camera_info", CameraInfo, self.left_info_callback)
        #rospy.Subscriber("/endoscope/right/camera_info", CameraInfo, self.right_info_callback)


    """
    def left_info_callback(self, msg):
        if self.info['l']:
            return
        self.info['l'] = msg
        f = open("calibration_data/camera_left.p", "w")
        pickle.dump(msg, f)
        f.close()


    def right_info_callback(self, msg):
        if self.info['r']:
            return
        self.info['r'] = msg
        f = open("calibration_data/camera_right.p", "w")
        pickle.dump(msg, f)
        f.close()


    def right_image_callback(self, msg):
        if rospy.is_shutdown():
            return
        self.right_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        scipy.misc.imsave('images/right' + str(self.rcounter) + '.jpg', self.right_image)
        self.rcounter += 1
    """


    def left_image_callback(self, msg):
        """ 
        Uses the utility methods to get processed patches, which can be directly 
        classified using a neural network. 
        """
        if rospy.is_shutdown():
            return
        self.left_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.lcounter += 1

        left_image_gray = utilities.rgb2gray(self.left_image)
        raw_size = (400,400)
        scaled_size = (32,32)
        stride = 100
        patches, centroids = utilities.get_processed_patches(left_image_gray, 
                                                             raw_size=raw_size,
                                                             scaled_size=scaled_size,
                                                             stride=stride,
                                                             save=True)
        pred_probs = self.model.predict(patches)
        predictions = np.argmax(pred_probs, axis=1)

        print("inside left_image_callback")
        print("original image and grayscale shapes {} and {}".format(
                self.left_image.shape, left_image_gray.shape))
        print("raw_size {}, stride {}".format(raw_size, stride))
        print("patches.shape = {}, centroids.shape = {}".format(
                patches.shape, centroids.shape))
        print("pred_probs.shape = {}, predictions.shape = {}".format(
                pred_probs.shape, predictions.shape))
        print("pred_probs:\n{}".format(pred_probs))
        print("whew ... now let's sleep for a few seconds\n")

        rospy.sleep(10)
        if (self.lcounter == 1):
            rospy.signal_shutdown("all done")


if __name__ == "__main__":
    a = DavinciMaster()
