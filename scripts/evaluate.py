"""
Use this for evaluating the peformance of the dvrk when asked to insert a needle
in some location, also at some specified angle.

For now, we'll pop-up the image 

(c) 2017/2018 by Daniel Seita
"""
import argparse
import cv2
import numpy as np
import os
import pickle
import sys
import utils as U
from autolab.data_collector import DataCollector
from dvrk.robot import *
np.set_printoptions(suppress=True, linewidth=180) 

# For the mouse callback method.
BACK_POINTS = [] 
FRONT_POINTS = []
image = None


def click(event, x, y, flags, param):
    global BACK_POINTS, FRONT_POINTS, image
             
    # If left mouse button clicked, record the starting (x,y) coordinates and
    # indicate that cropping is being performed. This is for the BACK point.
    if event == cv2.EVENT_LBUTTONDOWN:
        BACK_POINTS.append((x,y))
                                                 
    # Check to see if the left mouse button was released.
    elif event == cv2.EVENT_LBUTTONUP:
        FRONT_POINTS.append((x,y))
        idx = len(BACK_POINTS)
       
        # Draw a rectangle around the region of interest, w/center point.
        cv2.destroyAllWindows()
        thisname = "We're in the click method AFTER drawing the rectangle. (Press "+\
                   "any key to proceed, or ESC if I made a mistake.)"
        cv2.namedWindow(thisname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(thisname, 1800, 2600)
        cv2.rectangle(img=image,
                      pt1=BACK_POINTS[-1], 
                      pt2=FRONT_POINTS[-1], 
                      color=(0,0,255), 
                      thickness=2)
        cv2.circle(img=image,
                   center=BACK_POINTS[-1],
                   radius=6, 
                   color=(0,0,255), 
                   thickness=-1)
        cv2.circle(img=image,
                   center=FRONT_POINTS[-1],
                   radius=6, 
                   color=(0,0,255), 
                   thickness=-1)
        cv2.putText(img=image,
                    text="back {}: {}".format(idx, BACK_POINTS[-1]), 
                    org=BACK_POINTS[-1],  
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, 
                    color=(0,0,0), 
                    thickness=2)
        cv2.putText(img=image,
                    text="front {}: {}".format(idx, FRONT_POINTS[-1]), 
                    org=FRONT_POINTS[-1],  
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, 
                    color=(0,0,0), 
                    thickness=2)
        cv2.imshow(thisname, image)
        cv2.waitKey(0)


def specify_throws(args, d, which):
    """ 
    The user manually specifies how the throws should be done. Drag from the
    BACK to FRONT, for intuition. So the second click is where we want the
    needle tip to ENTER the phantom, and the first click is used to get the
    correct angle to which we want the needle to enter.

    Then we save the points, reset the list of points (that's why we need to
    fetch them with `global`) and do the same with the other camera image.

    BTW, if the user does this with a piece of paper that has lines, then we
    should just pull the piece of paper out once we have the correct desired
    points (if the paper is thin enough, the dvrk camera won't know about the
    height difference).
    """
    global BACK_POINTS, FRONT_POINTS, image
    if which == 'left':
        image = np.copy(d.left['raw'])
    elif which == 'right':
        image = np.copy(d.right['raw'])
    imdir = args.imdir+which
    num_throws = 0

    while num_throws < args.test:
        window_name = "Currently have {} throws out of {} planned. "+\
                "Click and drag the NEXT direction, NOW. Or press ESC to "+\
                "terminate the program.".format(num_throws, args.test)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, click) # Record clicks to this window!
        cv2.resizeWindow(window_name, 1800, 2600)
        U.call_wait_key(cv2.imshow(window_name, image))

        # Now save the image with the next available index.
        index = len(os.listdir(imdir))
        cv2.imwrite(imdir+"/point_"+str(index).zfill(2)+".png", image)
        cv2.destroyAllWindows()
        num_throws += 1
        
    assert len(BACK_POINTS) == len(FRONT_POINTS) == num_throws
    info = {'num_throws': num_throws,
            'back': BACK_POINTS,
            'front': FRONT_POINTS}
    BACK_POINTS = []
    FRONT_POINTS = []
    return info


def convert_for_robot(args, infol, infor):
    """ 
    Given left/right image info, convert this into stuff the dvrk can use
    and interpret. 
    """
    assert len(infol['back']) == len(infor['back'])
    assert len(infol['front']) == len(infor['front'])

    # TODO: convert these to "base" coordinates.

    # TODO: fit a line and get the possible candidate values for orientation

    # TODO: anything else?? Not clear what I'll put here or in `evaluate`.


def evaluate(args, info):
    """ Evaluate the performance of our needle thrower. """

    # TODO: um, just about everything ...

    pass


if __name__ == "__main__": 
    pp = argparse.ArgumentParser()
    pp.add_argument('--test', type=int, help='Number of throws to test.')
    pp.add_argument('--imdir', type=str, default='images/targets/')
    args = pp.parse_args()
    assert args.test >= 1

    arm1, arm2, d = U.init(sleep_time=2)
    arm1.close_gripper()

    # Step 1: User specifies the suturing throw directions.
    info_left  = specify_throws(args, d, which='left')
    info_right = specify_throws(args, d, which='right')

    # Step 2: Convert this information to dvrk-readable stuff.
    info = convert_for_robot(args, info_left, info_right)

    # Step 3: Run tests and evaluate.
    evaluate(args, info)
