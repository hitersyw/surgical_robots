"""
This module defines basic calibration primitives such as detecting 
the chessboard and localizing it

Update Feb 6 2016, I added in prints to help me follow the code. -Daniel Seita
"""

##Constants
ENDOSCOPE_LEFT_TOPIC = "/endoscope/left/"
ENDOSCOPE_RIGHT_TOPIC = "/endoscope/right/"
ENDOSCOPE_LEFT_RECT = ENDOSCOPE_LEFT_TOPIC + "image_rect_color"
ENDOSCOPE_RIGHT_RECT = ENDOSCOPE_RIGHT_TOPIC + "image_rect_color"
ENDOSCOPE_LEFT_INFO = ENDOSCOPE_LEFT_TOPIC + "camera_info"
ENDOSCOPE_RIGHT_INFO = ENDOSCOPE_RIGHT_TOPIC + "camera_info"
DVRK_PSM1 = '/dvrk/PSM1/'
DVRK_PSM2 = '/dvrk/PSM2/'

import os
import sys
import scipy.misc
import pickle
import image_geometry
import rospy
from geometry_msgs.msg import PointStamped, Point, PoseStamped
from visualization_msgs.msg import Marker
import cv2
import cv_bridge
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from sklearn.utils.extmath import cartesian
np.set_printoptions(suppress=True)


class StereoChessboardCollect(object):
  """
  This is the basic class that creates a instance of chessboard detection
  """

  def __init__(self, outputdir="./data/"):
    """
    To initialize a StereoChessboardCalibration takes an output directory
    """

    self.left_image = None
    self.right_image = None
    self.right_info = None
    self.left_info = None

    self.outputdir = outputdir
    self.bridge = cv_bridge.CvBridge()

    rospy.Subscriber(ENDOSCOPE_LEFT_RECT, Image, self._left_image_callback)
    rospy.Subscriber(ENDOSCOPE_RIGHT_RECT, Image, self._right_image_callback)
    rospy.Subscriber(ENDOSCOPE_LEFT_INFO, CameraInfo, self._left_image_info_callback)
    rospy.Subscriber(ENDOSCOPE_RIGHT_INFO, CameraInfo, self._right_image_info_callback)


  ##Internal methods

  def _left_image_callback(self, msg):
    """
    Internal method to handle left image callbacks
    """
    if rospy.is_shutdown():
      return
    self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

  def _right_image_callback(self, msg):
    """
    Internal method to handle right image callbacks
    """
    if rospy.is_shutdown():
      return
    self.right_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

  def _right_image_info_callback(self, msg):
    """
    Internal method to handle right camera info
    """
    if rospy.is_shutdown():
      return
    self.right_info = msg 

  def _left_image_info_callback(self, msg):
    """
    Internal method to handle left camera info
    """
    if rospy.is_shutdown():
      return
    self.left_info = msg 

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
        # both lists must be of the same lenghth otherwise return None
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

  ##external methods

  def getCheckerBoardCorners(self):
    """ 
    Daniel: I had to do a LOT of testing here for findChessboardCorners. 
    """
    n1,n2 = 9,7
    left_gray = cv2.cvtColor(self.left_image,cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(self.right_image,cv2.COLOR_BGR2GRAY)
    ret, left_corners = cv2.findChessboardCorners(left_gray, (n1,n2), flags=1)
    ret, right_corners = cv2.findChessboardCorners(right_gray, (n1,n2), flags=1)
    if left_corners is None:
      print("ERROR/WARNING: left_corners = None")
    if right_corners is None:
      print("ERROR/WARNING: right_corners = None")
    assert n1*n2 == left_corners.shape[0]
    assert n1*n2 == right_corners.shape[0]
    assert left_gray.shape == right_gray.shape
    left_gray_copy  = np.copy(left_gray)
    left_gray_new   = np.copy(left_gray)
    right_gray_copy = np.copy(right_gray)
    right_gray_new  = np.copy(right_gray)

    # Daniel: Thanks Carolyn!
    for c in left_corners:
      x,y = c.ravel()
      cv2.circle(left_gray, (x,y), radius=4, color=(255,255,255), thickness=2)
    for c in right_corners:
      x,y = c.ravel()
      cv2.circle(right_gray, (x,y), radius=4, color=(255,255,255), thickness=2)

    # Daniel: alternative is to use cv2.imwrite(...).
    scipy.misc.imsave(self.outputdir+"/"+'left_added_circles.jpg', left_gray)
    scipy.misc.imsave(self.outputdir+"/"+'right_added_circles.jpg', right_gray)
    np.save(self.outputdir+"/"+'left_chess_corners', left_corners)
    np.save(self.outputdir+"/"+'right_chess_corners', right_corners)

    # Now compute optimization problem solution for mapping one to another.
    # For P and Q, first column represents the *column index* within the image.
    # A is the solution for left-to-right, B is the solution for right-to-left.
    P = left_corners.reshape((n1*n2,2))
    Q = right_corners.reshape((n1*n2,2))

    # Solve for A. Note that we need a bias column in P.
    Pb = np.column_stack((P, np.ones(n1*n2)))
    a_col0 = (np.linalg.inv((Pb.T).dot(Pb)).dot(Pb.T)).dot(Q[:,0])
    a_col1 = (np.linalg.inv((Pb.T).dot(Pb)).dot(Pb.T)).dot(Q[:,1])
    A = np.column_stack((a_col0,a_col1))
    np.save(self.outputdir+"/"+'left_to_right_3x2', A)
    error_a = np.abs(Pb.dot(A)-Q)

    # Solve for B. Note that we need a bias column in Q.
    Qb = np.column_stack((Q, np.ones(n1*n2)))
    b_col0 = (np.linalg.inv((Qb.T).dot(Qb)).dot(Qb.T)).dot(P[:,0])
    b_col1 = (np.linalg.inv((Qb.T).dot(Qb)).dot(Qb.T)).dot(P[:,1])
    B = np.column_stack((b_col0,b_col1))
    np.save(self.outputdir+"/"+'right_to_left_3x2', B)
    error_b = np.abs(Qb.dot(B)-P)

    # Debug stuff.
    print("LEFT-TO-RIGHT:\nshape Pb = {}, Q = {}".format(Pb.shape, Q.shape))
    print("A:\n{}".format(A))
    print("Mean: {}, Median: {}".format(np.mean(error_a),np.median(error_a)))
    print("\nRIGHT-TO-LEFT:\nshape Qb = {}, P = {}".format(Qb.shape, P.shape))
    print("B:\n{}".format(B))
    print("Mean: {}, Median: {}".format(np.mean(error_b),np.median(error_b)))

    # Test corner mapping for both.
    for c in Pb:
      pt = (c.ravel()).dot(A)
      x,y = int(round(pt[0])), int(round(pt[1]))
      cv2.circle(left_gray, (x,y), radius=4, color=(0,0,0), thickness=2)
    scipy.misc.imsave(self.outputdir+"/"+'left_mapped_corners.jpg', left_gray)
    for c in Qb:
      pt = (c.ravel()).dot(B)
      x,y = int(round(pt[0])), int(round(pt[1]))
      cv2.circle(right_gray, (x,y), radius=4, color=(0,0,0), thickness=2)
    scipy.misc.imsave(self.outputdir+"/"+'right_mapped_corners.jpg', right_gray)

    # Finally, let's test it out. Uses a new function, `cartesian` to generate all 
    # possible pixel combinations. We may need thresholds to eliminate odd cases.
    numy,numx = left_gray.shape  # Should be (1080,1920), same for left/right.

    # Left-to-right mapping.
    coords_left = cartesian( (np.arange(200,1800),np.arange(100,1000)) )
    coords_left = np.column_stack( (coords_left, np.ones(coords_left.shape[0])) ).astype(int)
    coords_right = coords_left.dot(A).astype(int)
    print("\n(In left-to-right mapping)\ncoords_left.shape = {}".format(coords_left.shape))
    print("coords_right.shape = {}".format(coords_right.shape))
    left_gray_new = np.zeros((numy,numx))
    left_gray_new[ coords_right[:,1],coords_right[:,0] ] = left_gray_copy[ coords_left[:,1],coords_left[:,0] ]
    scipy.misc.imsave(self.outputdir+"/"+'left_mapped_to_right.jpg', left_gray_new)

    # Right-to-left mapping. It should override correctly...
    coords_right = cartesian( (np.arange(200,1800),np.arange(100,1000)) )
    coords_right = np.column_stack( (coords_right, np.ones(coords_right.shape[0])) ).astype(int)
    coords_left = coords_right.dot(B).astype(int) # It's 'B' NOT 'A'!!
    print("\n(In right-to-left mapping)\ncoords_right.shape = {}".format(coords_right.shape))
    print("coords_left.shape = {}".format(coords_left.shape))
    right_gray_new = np.zeros((numy,numx))
    right_gray_new[ coords_left[:,1],coords_left[:,0] ] = right_gray_copy[ coords_right[:,1],coords_right[:,0] ]
    scipy.misc.imsave(self.outputdir+"/"+'right_mapped_to_left.jpg', right_gray_new)

    # Whew!
    return (left_corners, right_corners)


  def saveOutputs(self):
    scipy.misc.imsave(self.outputdir+"/"+'left_raw.jpg', self.left_image)
    scipy.misc.imsave(self.outputdir+"/"+'right_raw.jpg', self.right_image)
    left, right = self.getCheckerBoardCorners()
    points = self._get_points_3d(left, right)
    pickle.dump((left,right), open(self.outputdir+"/"+"checkerboard.p","wb"))
    pickle.dump(points, open(self.outputdir+"/"+"endoscope_pts.p","wb"))



### computes a rigid transformation from a specified data directory

class RigidFrameTransformation(object):

  def __init__(self, arm, outputdir):
    self.outputdir = outputdir
    self.arm = arm.split('/')[2]
    self.robotPoints = None
    self.cameraPoints = None

  def loadFromDirectory(self):
    self.camera_matrix = np.load(open(self.outputdir+"/camera_matrix.p", "rb"))
    self.robot_matrix = np.load(open(self.outputdir+"/robot_matrix.p", "rb"))
    self.load_robot_points()
    self.load_camera_points()

  def load_robot_points(self):
    lst = []
    f3 = open(self.outputdir+"/"+self.arm+".p", "rb")
    pos1 = pickle.load(f3)
    lst.append(pos1)
    
    while True:
      try:
        pos2 = pickle.load(f3)
        lst.append(pos2)
      except:
        f3.close()
        self.robotPoints = np.matrix(lst)
        return self.robotPoints

    return self.robotPoints

  def load_camera_points(self):
    f3 = open(self.outputdir+"/endoscope_pts.p", "rb")
    pos1 = pickle.load(f3)
    self.cameraPoints = np.matrix([ (p.point.x, p.point.y, p.point.z) for p in pos1])
    return self.cameraPoints

  def saveOutputs(self):
    self.camera_matrix = self.solve_for_camera_matrix()
    self.robot_matrix = self.solve_for_robot_matrix()
    np.save(open(self.outputdir+"/camera_matrix.p", "wb"), self.camera_matrix)
    np.save(open(self.outputdir+"/robot_matrix.p", "wb"), self.robot_matrix)

  def solve_for_camera_matrix(self):
    """
    Returns Camera -> Robot frame matrix
    """
    robot_points = self.load_robot_points()
    camera_points = self.load_camera_points()
    camera_mean = camera_points.mean(axis=0)
    robot_mean = robot_points.mean(axis=0)

    for i in range(robot_points.shape[0]):
      robot_points[i,:] -= robot_mean
      camera_points[i,:] -= camera_mean

    X = camera_points.T
    Y = robot_points.T
    covariance = X * Y.T
    U, Sigma, V = np.linalg.svd(covariance)
    V = V.T
    idmatrix = np.identity(3)
    idmatrix[2, 2] = np.linalg.det(V * U.T)
    R = V * idmatrix * U.T
    t = robot_mean.T - R * camera_mean.T
    return np.concatenate((R, t), axis=1)

  def solve_for_robot_matrix(self):
    """
    Returns Robot -> Camera frame matrix
     """
    Trobot = np.zeros((4,4))
    Trobot[:3,:] = np.copy(self.camera_matrix)
    Trobot[3,3] = 1
    Rrobot = np.linalg.inv(Trobot)
    return Rrobot

  def camera2robot(self, u):
    if u.shape != (3,1):
      raise ValueError("The shape of the provided input is not (3,1)")

    pt = np.ones((4,1))
    pt[:3,:] = u

    return np.dot(self.camera_matrix, pt)

  def robot2camera(self, u):
    if u.shape != (3,1):
      raise ValueError("The shape of the provided input is not (3,1)")

    pt = np.ones((4,1))
    pt[:3,:] = u

    return np.dot(self.robot_matrix, pt)[:3]
