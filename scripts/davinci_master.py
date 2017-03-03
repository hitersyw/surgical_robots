"""
This is the master code and should be executed for our experiments.
The subscribers can save very quickly, but we'll make them sleep long enough.
Also, we need to import our neural networks code here.


# This is stuff from the right camera callback image.
self.rcounter += 1
right_image_gray = utilities.rgb2gray(self.right_image)
raw_size = (400,400)
scaled_size = (32,32)
stride = 100
patches, centroids = utilities.get_processed_patches(right_image_gray, 
                                                     raw_size=raw_size,
                                                     scaled_size=scaled_size,
                                                     stride=stride,
                                                     save=True,
                                                     left=False)
pred_probs = self.model.predict(patches)
predictions = np.argmax(pred_probs, axis=1)

# Map centers from left camera pixel locations to right camera pixel locations. Yeah, have to switch axes ...
centers_right = np.column_stack( (centroids[:,1], centroids[:,0], np.ones(centroids.shape[0])) ).astype(int)
centers_left = centers_right.dot(self.right_to_left).astype(int)
np.save(self.outfile+ "right_centroids", centers_right)
np.save(self.outfile+ "right_to_left_centroids", centers_left)

rospy.sleep(10)
if (self.rcounter == 1):
  rospy.signal_shutdown("all done")
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
import image_geometry

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
    self.left_image = None
    self.left_info = None
    self.right_image = None
    self.right_info = None
    self.rcounter = 0
    self.lcounter = 0
    self.bridge = cv_bridge.CvBridge()
    self.outfile = "misc/"

    # Load network and other information.
    self.model = load_model("networks/cnn_cifar_keras_style.h5")
    self.left_to_right = np.load("calibration_stuff/left_to_right_3x2.npy")
    self.right_to_left = np.load("calibration_stuff/right_to_left_3x2.npy")
    self.camera_matrix = np.load(open("calibration_stuff/camera_matrix.p", "rb"))
    self.robot_matrix  = np.load(open("calibration_stuff/robot_matrix.p", "rb"))

    rospy.init_node('image_saver') # Daniel: without this, self.left_image=None.
    #rospy.spin() # Daniel: not sure if this is needed

    #========SUBSCRIBERS========#
    rospy.Subscriber("/endoscope/left/image_rect_color", Image, self.left_image_callback, queue_size=1)
    rospy.Subscriber("/endoscope/right/image_rect_color", Image, self.right_image_callback, queue_size=1)
    rospy.Subscriber("/endoscope/left/camera_info", CameraInfo, self.left_info_callback)
    rospy.Subscriber("/endoscope/right/camera_info", CameraInfo, self.right_info_callback)


  def left_info_callback(self, msg):
    if rospy.is_shutdown():
      return
    self.left_info = msg


  def right_info_callback(self, msg):
    if rospy.is_shutdown():
      return
    self.right_info = msg


  def left_image_callback(self, msg):
    if rospy.is_shutdown():
      return
    self.left_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")


  def right_image_callback(self, msg):
    if rospy.is_shutdown():
      return
    self.right_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")


  def _stereo_to_disparity(self, u, v, disparity):
    stereoModel = image_geometry.StereoCameraModel()
    stereoModel.fromCameraInfo(self.left_info, self.right_info)
    (x,y,z) = stereoModel.projectPixelTo3d((u,v),disparity)
    cameraPoint = PointStamped()
    cameraPoint.header.frame_id = self.left_info.header.frame_id
    cameraPoint.header.stamp = rospy.Time.now()
    cameraPoint.point = Point(x,y,z)
    return cameraPoint


  def _get_points_3d(self, left_points, right_points):
    """ both lists must be of the same lenghth otherwise return None """
    if len(left_points) != len(right_points):
      print len(left_points), len(right_points)
      rospy.logerror("The number of left points and the number of right points is not the same")
      return None

    points_3d = []
    for i in range(len(left_points)):
      a = np.ravel(left_points[i])
      b = np.ravel(right_points[i])
      disparity = abs(a[0]-b[0])
      pt = self._stereo_to_disparity(a[0], a[1], disparity)
      points_3d.append(pt)
    return points_3d


  def predict_left_camera_images(self):
    """ 
    Uses the utility methods to get processed patches, which can be directly 
    classified using a neural network. 
    """
    print("inside left camera images prediction method")
    left_image_gray = utilities.rgb2gray(self.left_image)
    raw_size = (100,100)
    scaled_size = (32,32)
    stride = 100
    patches, centroids = utilities.get_processed_patches(left_image_gray, 
                                                         raw_size=raw_size,
                                                         scaled_size=scaled_size,
                                                         stride=stride,
                                                         save=True,
                                                         left=True)
    pred_probs = self.model.predict(patches)
    predictions = np.argmax(pred_probs, axis=1)
    print("pred_probs.shape = {}".format(pred_probs.shape))
    for row in range(pred_probs.shape[0]):
      if (pred_probs[row,1] > pred_probs[row,0]):
        print("predictions in row {}: {}".format(row, pred_probs[row,:]))

    # Map centers from left camera pixel locations to right camera pixel locations. Yeah, have to switch axes ...
    # Doing np.max(centers_left, axis=0) gives for instance 1600 for first, 800 for second.
    # My interpretation: (x,y) for centers_left means we go right x pixels, down y pixels, when viewing the image.
    # This is intuitive but only works because I've explicitly switched columns with centroids.
    # WAIT we don't want that extra column for centers_left after we do the matrix multiplication ...
    centers_left = np.column_stack( (centroids[:,1], centroids[:,0], np.ones(centroids.shape[0])) ).astype(int)
    centers_right = centers_left.dot(self.left_to_right).astype(int)
    centers_left = np.column_stack( (centroids[:,1], centroids[:,0]) )
    np.save(self.outfile+ "left_centroids", centers_left)
    np.save(self.outfile+ "left_to_right_centroids", centers_right)

    # OK now aligning the points. I assume points3d is for the camera perspective.
    points3d = self._get_points_3d(centers_left, centers_right)
    cpoints = np.matrix([ (p.point.x, p.point.y, p.point.z) for p in points3d])
    print("len(points3d): {}".format(len(points3d)))
    print("cpoints.shape = {}".format(cpoints.shape))
    print("centers_left.shape = {}, centers_right.shape = {}".format(centers_left.shape, centers_right.shape))

    # Thus, now do camera --> robot.
    for i,c in enumerate(cpoints):
      print("point {}, predicted robot point = {}".format(i, self.camera2robot(c.T).T))


  def camera2robot(self, u):
    """ 
    Copied from Sanjay's method, should be OK since we pre-load our camera matrix. 
    """
    if u.shape != (3,1):
      raise ValueError("The shape of the provided input is not (3,1)")
    pt = np.ones((4,1))
    pt[:3,:] = u
    return np.dot(self.camera_matrix, pt)


if __name__ == "__main__":
    a = DavinciMaster()
    rospy.sleep(2) # Sanjay had this here, not sure why
    a.predict_left_camera_images() # well, it does everything basically =)
