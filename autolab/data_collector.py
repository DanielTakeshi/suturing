"""
The DataCollector class polls data from the rostopics periodically. It manages
the messages that come from ros.
"""
import numpy as np
import rospy
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker
import cv2
import cv_bridge
import numpy as np
import scipy.misc
import pickle
import imutils
import time
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
import string
import random


class DataCollector:

    def __init__(self, 
                 camera_left_topic="/endoscope/left/",
                 camera_right_topic="/endoscope/right/",
                 camera_info_str='camera_info',
                 camera_im_str='image_rect_color'):

        # Images get saved much faster if this is False.
        self.preprocess = False

        # Left and right images, and processed images.
        self.left = {}
        self.right = {}

        # Bounding box of points, to filter away any nonsense.
        self.lx, self.ly, self.lw, self.lh = 675, 300, 750, 750
        self.rx, self.ry, self.rw, self.rh = 600, 300, 750, 750
        self.left_apply_bbox  = True
        self.right_apply_bbox = True

        # Other stuff determined from the cameras.
        self.right_contours = []
        self.left_contours = []

        self.info = {'l': None, 'r': None}
        self.bridge = cv_bridge.CvBridge()
        self.timestep = 0
        self.identifier = ''.join(
                random.choice(string.ascii_uppercase + string.digits) for _ in range(20))

        rospy.Subscriber(camera_left_topic + camera_im_str, Image,
                         self.left_image_callback, queue_size=1)
        rospy.Subscriber(camera_right_topic + camera_im_str, Image,
                         self.right_image_callback, queue_size=1)
        rospy.Subscriber(camera_left_topic + camera_info_str,
                         CameraInfo, self.left_info_callback)
        rospy.Subscriber(camera_right_topic + camera_info_str,
                         CameraInfo, self.right_info_callback)
        self.start_t = time.time()


    def left_info_callback(self, msg):
        if self.info['l']:
            return
        self.info['l'] = msg


    def right_info_callback(self, msg):
        if self.info['r']:
            return
        self.info['r'] = msg


    def right_image_callback(self, msg):
        if rospy.is_shutdown():
            return
        x,y,w,h = self.rx, self.ry, self.rw, self.rh
        sr = self.right

        sr['raw'] = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        if self.preprocess:
            sr['bbox'] = self.make_bounding_box(sr['raw'], x,y,w,h)
            sr['bin_mask'] = self._binary_mask(sr['raw'])
            sr['proc_default'] = self._preproc_default(sr['raw'])
            self.right_contours = self._get_contours(sr['proc_default'])


    def left_image_callback(self, msg):
        print("in left_image_callback, time: {}".format(time.time() - self.start_t))
        if rospy.is_shutdown():
            return
        x,y,w,h = self.lx, self.ly, self.lw, self.lh
        sl = self.left

        sl['raw'] = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        if self.preprocess:
            sl['bbox'] = self.make_bounding_box(sl['raw'], x,y,w,h)
            sl['bin_mask'] = self._binary_mask(sl['raw'])
            sl['proc_default'] = self._preproc_default(sl['raw'])
            self.left_contours = self._get_contours(sl['proc_default'])

    ###########
    # Helpers #
    ###########

    def get_left_bounds(self):
        return self.lx, self.ly, self.lw, self.lh
    def get_right_bounds(self):
        return self.rx, self.ry, self.rw, self.rh

    
    def _binary_mask(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img.copy(), 127, 255, cv2.THRESH_BINARY)
        return thresh


    def _preproc_default(self, img):
        img = cv2.medianBlur(img, 9)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.bilateralFilter(img, 7, 13, 13)
        return cv2.Canny(img,100,200)


    def make_bounding_box(self, img, x,y,w,h):
        """ Make a bounding box. """
        img = img.copy()
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,0), 2)
        return img


    def _get_contours(self, img):
        (cnts, _) = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
        processed_countours = []

        for c in cnts:
            try:
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                processed_countours.append((cX, cY, approx, peri))
            except:
                pass

        processed_countours = sorted(processed_countours, key = lambda x: x[0])
        processed_countours = sorted(processed_countours, key = lambda x: x[1])
        return processed_countours
