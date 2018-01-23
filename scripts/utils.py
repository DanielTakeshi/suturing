""" For reducing clutter. """
from autolab.data_collector import DataCollector
from dvrk.robot import robot
import cv2
import image_geometry
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
camerainfo_p = '/home/davinci0/danielseita/suturing/config/camera_info_matrices/'
C_LEFT_INFO  = pickle.load(open(camerainfo_p+'left.p',  'r'))
C_RIGHT_INFO = pickle.load(open(camerainfo_p+'right.p', 'r'))
STEREO_MODEL = image_geometry.StereoCameraModel()
STEREO_MODEL.fromCameraInfo(C_LEFT_INFO, C_RIGHT_INFO)


def camera_pixels_to_camera_coords(left_pt, right_pt):
    """ Given [lx,ly] and [rx,ry], determine [cx,cy,cz]. Everything should be LISTS. """
    assert len(left_pt) == len(right_pt) == 2
    disparity = np.linalg.norm( np.array(left_pt) - np.array(right_pt) )
    (xx,yy,zz) = STEREO_MODEL.projectPixelTo3d( (left_pt[0],left_pt[1]), disparity )
    return [xx, yy, zz] 


def init(sleep_time=0):
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
    if pos[2] < -0.17:
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


def pos_rot_cpos(cpos, nparrays=False):
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


def pos_rot_arm(arm, nparrays=False):
    """ 
    Since I don't want to keep tying the get_current_car ...  Also, using the
    nparrays means printing is typically easier, with fewer decimal places.
    """
    return pos_rot_cpos(arm.get_current_cartesian_position(), nparrays)


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
    """  Load from pickle file in a list, helpful if we've appended a lot.  """
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


# ----------------------------------------
# Rigid Transformation (or 'registration')
# ----------------------------------------


def solve_rigid_transform(X, Y, debug=True):
    """
    Takes in two sets of corresponding points, returns the rigid transformation
    matrix from the FIRST TO THE SECOND. This is a (3,4) matrix so we'd apply it
    on the original points in their homogeneous form with the fourth coordinate
    equal to one. This is slightly different from Brijen's code since I'm using
    np.array, not np.matrix.

    Notation: A for camera points, B for robot points, so want to find an affine
    mapping from A -> B with orthogonal rotation and a translation. This means
    the matrix P here would be written in math as P_{ba}, so b = Pa where a and
    b are vectors wrt A and B, respectively. 
    
    Technically, both a and b should be 4D vectors but we're cheating a bit by
    leaving the last row of P out ... it's a (3,4) matrix, not a (4,4) matrix,
    so b is a 3D vector, not 4D.
    """
    assert X.shape[0] == Y.shape[0] >= 3
    assert X.shape[1] == Y.shape[1] == 3
    A = X.T  # (3,N)
    B = Y.T  # (3,N)

    # Look for Inge Soderkvist's solution online if confused.
    meanA = np.mean(A, axis=1, keepdims=True)
    meanB = np.mean(B, axis=1, keepdims=True)
    A = A - meanA
    B = B - meanB
    covariance = B.dot(A.T)
    U, sigma, VH = np.linalg.svd(covariance) # VH = V.T, i.e. numpy transposes it for us.

    V = VH.T
    D = np.eye(3)
    D[2,2] = np.linalg.det( U.dot(V.T) )
    R = U.dot(D).dot(V.T)
    t = meanB - R.dot(meanA)
    RB_matrix = np.concatenate((R, t), axis=1)

    #################
    # SANITY CHECKS #
    #################

    print("\nBegin debug prints for rigid transformation from A to B:")
    print("meanA:\n{}\nmeanB:\n{}".format(meanA, meanB))
    print("Rotation R:\n{}\nand R^TR (should be identity):\n{}".format(R, (R.T).dot(R)))
    print("translation t:\n{}".format(t))
    print("RB_matrix:\n{}".format(RB_matrix))

    # Get residual to inspect quality of solution. Use homogeneous coordinates for A.
    # Also, recall that we're dealing with (3,N) matrices, not (N,3).
    # In addition, we don't want to zero-mean for real applications.
    A = X.T # (3,N)
    B = Y.T  # (3,N)

    ones_vec = np.ones((1, A.shape[1]))
    A_h = np.concatenate((A, ones_vec), axis=0)
    B_pred = RB_matrix.dot(A_h)
    assert B_pred.shape == B.shape

    # Careful! Use raw_errors for the RF, but it will depend on pred-targ or targ-pred.
    raw_errors = B_pred - B # Use pred-targ, of shape (3,N)
    l2_per_example = np.sum((B-B_pred)*(B-B_pred), axis=0)
    frobenius_loss = np.mean(l2_per_example)

    if debug:
        print("\nInput, A.T:\n{}".format(A.T))
        print("Target, B.T:\n{}".format(B.T))
        print("Predicted points:\n{}".format(B_pred.T))
        print("Raw errors, B-B_pred:\n{}".format((B-B_pred).T))
        print("Mean abs error per dim: {}".format( (np.mean(np.abs(B-B_pred), axis=1))) )
        print("Residual (L2) for each:\n{}".format(l2_per_example.T))
        print("loss on data: {}".format(frobenius_loss))
        print("End of debug prints for rigid transformation.\n")

    assert RB_matrix.shape == (3,4)
    return RB_matrix
