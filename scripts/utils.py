""" For reducing clutter. """

from autolab.data_collector import DataCollector
from dvrk.robot import robot
import cv2
# import image_geometry # ?? Why does this not work anymore?
import numpy as np
import os
import pickle
import sys
import tfx
import time
np.set_printoptions(suppress=True)


# ------------
# dVRK-related
# ------------

ESC_KEYS     = [27, 1048603]
## C_LEFT_INFO  = pickle.load(open('config/camera_info_matrices/left.p',  'r'))
## C_RIGHT_INFO = pickle.load(open('config/camera_info_matrices/right.p', 'r'))
## STEREO_MODEL = image_geometry.StereoCameraModel()
## STEREO_MODEL.fromCameraInfo(C_LEFT_INFO, C_RIGHT_INFO)


## def camera_pixels_to_camera_coords(left_pt, right_pt):
##     """ Given [lx,ly] and [rx,ry], determine [cx,cy,cz]. Everything should be LISTS. """
##     assert len(left_pt) == len(right_pt) == 2
##     disparity = np.linalg.norm( np.array(left_pt) - np.array(right_pt) )
##     (xx,yy,zz) = STEREO_MODEL.projectPixelTo3d( (left_pt[0],left_pt[1]), disparity )
##     return [xx, yy, zz] 


## def home(arm1=None, arm2=None, rot1=None, rot2=None):
##     """ No more `arm.home()` calls!!! Handles both arms simultaneously. """
##     assert (arm1 is not None) or (arm2 is not None), "Error, both arms are none"
##     rot = None
##     rot = None
##     if rot is not None:
##         rot = None
##         rot = None
##     #move(arm1, pos=[0.00, 0.06, -0.15], rot=rot, SPEED_CLASS='Fast')
##     #move(arm2, pos=[0.00, 0.06, -0.15], rot=rot, SPEED_CLASS='Fast')
##     #else:
##     #    move(arm1, pos=[0.00, 0.06, -0.15], rot=[0,10,-165], SPEED_CLASS='Fast')
##     #    move(arm2, pos=[0.00, 0.06, -0.15], rot=[0,10,-165], SPEED_CLASS='Fast')


def init(sleep_time=2):
    """ This is often used in the start of every script. """
    d = DataCollector()
    r1 = robot("PSM1") # left (but my right)
    r2 = robot("PSM2") # right (but my left)
    time.sleep(sleep_time)
    return (r1,r2,d)


def move(arm, pos, rot, speed='Slow'):
    """ Handles the different speeds we're using.
    
    Parameters
    ----------
    arm: [dvrk arm]
        The current DVK arm.
    pos: [list]
        The desired position.
    rot: [list]
        The desired rotation, in list form with yaw, pitch, and roll.
    SPEED_CLASS: [String]
        Slow, Medium, or Fast.
    """
    if pos[2] < -0.18:
        raise ValueError("Desired z-coord of {} is not safe! Terminating!".format(pos[2]))
    if speed == 'Slow':
        arm.move_cartesian_frame_linear_interpolation(
                tfx.pose(pos, tfx.tb_angles(rot[0],rot[1],rot[2])), 0.03
        )
    elif speed == 'Medium':
        arm.move_cartesian_frame_linear_interpolation(
                tfx.pose(pos, tfx.tb_angles(rot[0],rot[1],rot[2])), 0.06
        )
    elif speed == 'Fast':
        arm.move_cartesian_frame(tfx.pose(pos, tfx.tb_angles(rot[0],rot[1],rot[2])))
    else:
        raise ValueError()


def get_pos_rot_from_cpos(cpos, nparrays=False):
    """
    It's annoying to have to do this every time. I just want two lists.
    To be clear, `cpos = arm.get_current_cartesian_position()`.
    """
    pos  = np.squeeze(np.array(cpos.position[:3])).tolist()
    rott = tfx.tb_angles(cpos.rotation)
    rot  = [rott.yaw_deg, rott.pitch_deg, rott.roll_deg]
    assert len(pos) == len(rot) == 3
    if nparrays:
        pos = np.array(pos)
        rot = np.array(rot)
    return pos, rot


def get_pos_rot_from_arm(arm, nparrays=False):
    """ Since I don't want to keep tying the get_current_car ... """
    return get_pos_rot_from_cpos(arm.get_current_cartesian_position(), nparrays)


def filter_point(x, y, xlower, xupper, ylower, yupper):
    """ 
    Used in case we want to filter out contours that aren't in some area. 
    Returns True if we _should_ ignore the point.
    """
    ignore = False
    if (x < xlower or x > xupper or y < ylower or y > yupper):
        ignore = True
    return ignore


# ------------
# Data Storage 
# ------------


def store_pickle(fname, info, mode='w'):
    """ Overrides (due to `w`) by default. """
    assert fname[-2:] == '.p'
    f = open(fname, mode)
    pickle.dump(info, f)
    f.close()


def load_pickle_to_list(filename, squeeze=True):
    """ 
    Load from pickle file in a list, helpful if we've appended lots of stuff. 
    """
    f = open(filename,'r')
    data = []
    while True:
        try:
            data.append(pickle.load(f))
        except EOFError:
            break
    assert len(data) >= 1
    f.close()
    
    # if length is one, might as well just 'squeeze' it...
    if len(data) == 1 and squeeze:
        return data[0]
    else:
        return data
 

def call_wait_key(nothing=None, exit=True):
    """ Call this like: `utils.call_wait_key( cv2.imshow(...) )`. 
    
    Normally, pressing ESC just terminates the program with `sys.exit()`. However
    in some cases I want the program to continue and the ESC was just to get it
    to exit a loop. In that case, set `exit=False`.
    """
    ESC_KEYS = [27, 1048603]
    key = cv2.waitKey(0)
    if key in ESC_KEYS:
        if exit:
            print("Pressed ESC key. Terminating program...")
            sys.exit()
        else:
            print("Pressed ESC key, but NOT exiting ... ")
    return key


def save_images(d, image_dir='images/'):
    """ For debugging/visualization of DataCollector. """
    for (lkey, rkey) in zip(d.left, d.right):
        cv2.imwrite(image_dir+lkey+"_left.png", d.left[lkey])
        cv2.imwrite(image_dir+rkey+"_right.png", d.right[rkey])
