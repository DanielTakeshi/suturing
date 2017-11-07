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


def iterate_through_contours(image_raw, image_c, bb):
    """ 
    Use `image_raw` for the raw image and `image_c` for what we want to take
    contours w.r.t. Use `image_raw` for plotting.
    """
    name = "Here's where we searching for contours:"
    cv2.imshow(name, image_c) 
    key = cv2.waitKey(0) 
    cv2.destroyAllWindows()

    (cnts, _) = cv2.findContours(image_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    print("number of contours: {}".format(len(cnts)))
    ellipses_for_later = []
    image_raw_later = image_raw.copy() # A clean copy in case we need it.

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

            cv2.ellipse(image_raw, ellipse, (0,255,0), 2)
            name = "Is this ellipse good?  ESC to skip it, else add it."
            cv2.namedWindow(name, cv2.WINDOW_NORMAL) 
            cv2.resizeWindow(name, 2000, 4000)
            cv2.imshow(name, image_raw) 

            firstkey = cv2.waitKey(0) 
            if firstkey not in U.ESC_KEYS:
                ellipses_for_later.append(ellipse)
        except:
            pass


def detect_via_ellipse(d, image, bb):
    """ 
    Unfortunately, this doesn't seem to work well because the contours don't
    actually detect the full needle itself w/out too much extra, even with the
    default preprocessing ...

    The `image` here is the raw LEFT image.
    """
    iterate_through_contours(image_raw=image, image_c=d.left['proc_default'], bb=bb)


def detect_via_binary_masks(img, bb):
    """ Might be closer to what Sanjay did ... 
    
    OK this is going a little better than the contour detection ... but I still
    need to figure out a way to identify the needle tip.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresholds = [60,70,80,90,100,110,120,130,140]
    images = []

    for th in thresholds:
        ret, thresh1 = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
        images.append(thresh1)

    for th,image in zip(thresholds,images):
        cv2.imshow("binarized image w/threshold {}".format(th), image)
        firstkey = cv2.waitKey(0) 
        cv2.destroyAllWindows()
        image = cv2.medianBlur(image, 9)
        image = cv2.bilateralFilter(image, 7, 13, 13)
        cv2.imshow("binarized image, thresh {}, AFTER fliter/blur".format(th), image)
        firstkey = cv2.waitKey(0) 
        cv2.destroyAllWindows()
   

def detect_via_color_segm(img, bb):
    """ 
    Maybe if the background is all the same, we can mask it out and leave the
    needle? Of course this still raises the question of _how_ to detect needle
    end-points... or could we use ANOTHER deep network for that? Gaaah. And I
    dont expect this to work that well because the green background varies a lot
    when it comes to the endoscopic camera's annoying lights.

    BTW our background is that green-ish foam.
    """
    hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
    cv2.imshow("here\'s the hsv image", hsv)
    key = cv2.waitKey(0)

    # Blue is supposed to have H=120 or so, green for H=60 ish.
    boundaries = [
            #([50,20,20], [70,255,255]),
            #([40,20,20], [80,255,255]),
            ([30,20,20], [90,255,255]),
            ([20,20,20], [100,255,255]),
            ([10,20,20], [110,255,255]),
            ([ 0,20,20], [120,255,255]),
            #([50,50,50], [70,255,255]),
            #([40,50,50], [80,255,255]),
            ([30,50,50], [90,255,255]),
            ([20,50,50], [100,255,255]),
            ([10,50,50], [110,255,255]),
            ([ 0,50,50], [120,255,255]),
            #([50,100,100], [70,255,255]), # Don't use 
            #([40,100,100], [80,255,255]),
            #([30,100,100], [90,255,255]),
            #([20,100,100], [100,255,255]),
            #([10,100,100], [110,255,255]),
            #([ 0,100,100], [120,255,255]),
            ([30,50,50], [170,255,255]),
            ([20,50,50], [180,255,255]),
            ([10,50,50], [190,255,255]),
            ([ 0,50,50], [200,255,255]),
    ]

    for (lower, upper) in boundaries:
        image = hsv.copy()
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
                     
        # find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        #name = "current mask for lower={}, upper={}".format(lower, upper)
        #cv2.imshow(name, output)
        #key = cv2.waitKey(0)
        name = "resulting masked image, lower={}, upper={}".format(lower, upper)
        cv2.imshow(name, output)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Try to process the image.
        out = cv2.medianBlur(output, 9)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.bilateralFilter(out , 7, 13, 13)
        out = cv2.Canny(out, 100, 200) 

        name = "for that previous masked image, here is it processed"
        cv2.imshow(name, out)
        key = cv2.waitKey(0)


def detect_contours_known_setting(img, bb, index):
    """ If I _know_ any settings that are good, might check out contours here. """

    if index == 0:
        hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
        cv2.imshow("here\'s the hsv image", hsv)
        key = cv2.waitKey(0)

        lower, upper = ([ 0,50,50], [200,255,255])
        image = hsv.copy()
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
                     
        # find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        # Try to process the image.
        out = cv2.medianBlur(output, 9)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.bilateralFilter(out , 7, 13, 13)
        out = cv2.Canny(out, 100, 200) 

    elif index == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, out = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
        out = cv2.medianBlur(out, 9)
        out = cv2.bilateralFilter(out, 7, 13, 13)
    
    else:
        raise ValueError()

    iterate_through_contours(image_raw=img, image_c=out, bb=bb)


if __name__ == "__main__":
    arm1, arm2, d = U.init()
    U.save_images(d)

    #for i in range(5):
    #    print("Now opening and closing ...")
    #    arm1.open_gripper(40)
    #    time.sleep(2)
    #    arm1.close_gripper()
    #    time.sleep(2)

    bb = d.get_left_bounds()
    image = d.left['raw'].copy()

    # Try different methods.
    #detect_via_ellipse(d, image.copy(), bb)
    #detect_via_binary_masks(image.copy(), bb)
    detect_via_color_segm(image.copy(), bb)
    #detect_contours_known_setting(image.copy(), bb, index=1)
