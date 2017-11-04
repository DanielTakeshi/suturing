"""
Check to see if the dVRK can perceive the needles. 
"""

from autolab.data_collector import DataCollector
from dvrk import robot
import argparse
import cv2
import numpy as np
import os
import pickle
import sys
import tfx
import time
import utils as U
np.set_printoptions(suppress=True, precision=5)


if __name__ == "__main__":
    arm1, arm2, d = U.init()
    arm1.open_gripper(10)
    arm1.close_gripper()
    U.save_images(d)

    (cnts, _) = cv2.findContours(d.left['proc_default'], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    print("number of contours: {}".format(len(cnts)))
    bb = d.get_left_bounds()
    image = d.left['raw'].copy()

    # Now try and detect needles.
    for c in cnts:
        try:
            # Find the centroids of the contours in _pixel_space_. :)
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if U.filter_point(cX,cY,xlower=bb[0],xupper=bb[0]+bb[2],ylower=bb[1],yupper=bb[1]+bb[3]):
                continue

            ellipse = cv2.fitEllipse(c)

            print("passed the if statement")
            print(cX,cY)
            print("here's the ellipse: {}".format(ellipse))

            cv2.ellipse(image, ellipse, (0,255,0), 2)
            name = "Is this ellipse good?  ESC to skip it, else add it."
            cv2.namedWindow(name, cv2.WINDOW_NORMAL) 
            cv2.resizeWindow(name, 2000, 4000)
            cv2.imshow(name, image) 
            firstkey = cv2.waitKey(0) 

        except:
            pass
