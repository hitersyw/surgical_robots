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

# Some stuff needed for Keras, etc.
from keras.models import load_model


class DavinciMaster:

    def __init__(self):
        self.right_image = None
        self.left_image = None
        self.rcounter = 0
        self.lcounter = 0
        self.info = {'l': None, 'r': None}
        self.bridge = cv_bridge.CvBridge()

        # Network loading (TODO file obviously has to change to be the correct one ...)
        self.model = load_model("networks/cnn_cifar_keras_style.h5")

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
        if rospy.is_shutdown():
            return
        self.left_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.lcounter += 1
        result = predict(self.left_image) # have each camera call this?

        # TODO do some stuff with result here ...
        # TODO then manage logic about how fast and long we want to run 

        rospy.sleep(2)
        if (self.lcounter == 2):
            rospy.signal_shutdown("all done")


    def predict(self, img):
        """ prediction code (draft), returns 0=NORMAL or 1=DEFORMED. """
        sh0, sh1, sh2 = img.shape
        img = img.reshape(sh0, sh1, sh2, 1)
        img /= 255
        return np.argmax(self.model.predict(img), axis=1)


if __name__ == "__main__":
    a = DavinciMaster()
