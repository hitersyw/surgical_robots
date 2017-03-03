#!/usr/bin/python

from calibration import *
import rospy

#Collects Chessboard data
# Daniel: I see, this uses the endoscope in a similar way as I did, by using rospy Subscribers which are focused on a chessboard.
# Upon initialization set up the output directory, then set up the four usual (two image, two camera) Susbcribers.
# The methods in the subscribers don't seem to do much, just saves camera image or or some `msg` (?) to local variables.
# Then saves the left/right checkboard images (I saw those) *and* gets the corners. Not sure how that works.
# Same as Brijen's code, saves left/right points (25 of them, btw). **We use those to compute the camera->robot transformation and vice versa.**
# It uses stereo to disparity code for this purpose; how does that work? Also should I have to redo this?

rospy.init_node('calibration')
s = StereoChessboardCollect("tests/")
rospy.sleep(2)
s.saveOutputs()
