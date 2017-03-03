"""
Testing davinci.
-Daniel Seita
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
import matplotlib.pyplot as plt
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

        self.right_patches = None
        self.left_patches = None

        #========SUBSCRIBERS========#
        # image subscribers
        rospy.init_node('image_saver')
        rospy.Subscriber("/endoscope/left/image_rect_color", Image, self.left_image_callback, queue_size=1)
        rospy.Subscriber("/endoscope/right/image_rect_color", Image, self.right_image_callback, queue_size=1)
        # info subscribers
        rospy.Subscriber("/endoscope/left/camera_info", CameraInfo, self.left_info_callback)
        rospy.Subscriber("/endoscope/right/camera_info", CameraInfo, self.right_info_callback)
        rospy.spin()

        print("After rospy.spin() shut down")
        print("\tleft_patches.shape = {}".format(self.left_patches.shape))
        print("\tright_patches.shape = {}".format(self.right_patches.shape))
        print("right_image.shape = {}, left_image.shape = {}".format(self.right_image.shape, self.left_image.shape))

        #======= SAVE IMAGES and MEASURE DISPARITY =======#
        cv2.imwrite("color_left.jpg", self.left_image)
        cv2.imwrite("color_right.jpg", self.right_image)
        stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET, ndisparities=16, SADWindowSize=15)
        img1 = cv2.cvtColor(self.left_image, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(self.right_image, cv2.COLOR_BGR2GRAY)
        disparity = stereo.compute(img1, img2)
        cv2.imwrite("disparity.jpg", disparity)


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
        self.analyze_images(self.right_image, camera="right")
        print("Analyzed image on right arm")
        self.rcounter += 1
        rospy.sleep(5)
        if (self.rcounter == 1):
            rospy.signal_shutdown("all done")


    def left_image_callback(self, msg):
        if rospy.is_shutdown():
            return
        self.left_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.analyze_images(self.left_image, camera="left")
        print("Analyzed image on left arm")
        self.lcounter += 1
        rospy.sleep(5)
        if (self.lcounter == 1):
            rospy.signal_shutdown("all done")


    def analyze_images(self, image, camera):
        """ Hopefully each arm can do this. """
        gray_img = utilities.rgb2gray(image)
        raw_size = (400,400)
        scaled_size = (32,32)
        stride = 100
        p,c = utilities.get_patches(gray_img, 
                                    raw_size=raw_size,
                                    scaled_size=scaled_size,
                                    stride=stride,
                                    save=False)
        if (camera == "left"):
            self.left_patches = p
            #self.left_image = gray_img
        else:
            self.right_patches = p
            #self.right_image = gray_img


if __name__ == "__main__":
    a = DavinciMaster()
