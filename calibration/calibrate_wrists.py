"""
This is for calibration, but using 'wrist calibration' so that the camera pixels
corresond to roughly where the wrists are located, NOT where the end-effector
TIPS are located, which is what I had before. The grid has 25 spots, which we
better indicate via clicking. Keep the brightness small at 10%.

Usage: `python calibration/calibrate_wrists.py`

Steps:

    - Get the set up correctly with the rectangular grid with cells.
    - Run the code and explicitly check contours. (Edit: actually, I'm going to
      do this by explicitly indicating the points with my click method, as it's
      way too hard to see otherwise.)
    - Once all contours have been selected, iterate through them.
    - For each, manually put the left end-effector *inside* the gripper. Then
      it's as close to mapping from pixels to wrist location as possible. 
    - Repeat immediately for the same point using the *right* end-effector.

Reminders:

    - Keep the surgical cameras and phantom and calibration setup FIXED.
    - Do ONE calibration per contour, to avoid confusion.
    - I might want to push to GitHub as soon as possible.

TL;DR: this code gives us data from (left_camera, right_camera) --- which we can
easily map to what are known as 'camera frame points' --- to positions w.r.t.
the base frame. I think it's OK if we use both arms as for each wrist that gives
us double the number of data points?
"""
from dvrk.robot import robot
from autolab.data_collector import DataCollector
from scripts import utils as U
from collections import defaultdict
import argparse, cv2, pickle, os, sys
import numpy as np

# For the mouse callback method, dragging boxes on images.
POINTS          = []
CENTER_OF_BOXES = []
image = None


def debug_images(d):
    """ Keep this here in case I need to debug images. """
    cv2.imshow("Left Camera Image (debugging only, DON'T CLICK!)", d.left['raw'])
    cv2.waitKey(0)
    cv2.imshow("Right Camera Image (debugging only, DON'T CLICK!)", d.right['raw'])
    cv2.waitKey(0)
    cv2.imwrite('calibration/calib_left.png', d.left['raw'])
    cv2.imwrite('calibration/calib_right.png', d.right['raw'])


def debug_info(info_left, info_right):
    print("\tLeft:")
    print("points:  {}".format(info_left['points']))
    print("centers: {}".format(info_left['centers']))
    print("\tRight:")
    print("points:  {}".format(info_right['points']))
    print("centers: {}".format(info_right['centers']))


def click(event, x, y, flags, param):
    global POINTS, CENTER_OF_BOXES, image
             
    # If left mouse button clicked, record the starting (x,y) coordinates and
    # indicate that cropping is being performed.
    if event == cv2.EVENT_LBUTTONDOWN:
        POINTS.append((x,y))
                                                 
    # Check to see if the left mouse button was released.
    elif event == cv2.EVENT_LBUTTONUP:
        POINTS.append((x,y))

        upper_left  = POINTS[-2]
        lower_right = POINTS[-1]
        assert upper_left[0] < lower_right[0]
        assert upper_left[1] < lower_right[1]
        center_x = int(upper_left[0] + (lower_right[0]-upper_left[0])/2)
        center_y = int(upper_left[1] + (lower_right[1]-upper_left[1])/2)
        CENTER_OF_BOXES.append( (center_x,center_y) )

        cv2.destroyAllWindows()
        thisname = "We're in the click method AFTER drawing the rectangle. (Press "+\
                   "any key to proceed, or ESC if I made a mistake.)"
        cv2.namedWindow(thisname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(thisname, 1800, 2600)
        cv2.rectangle(img=image,
                      pt1=POINTS[-2], 
                      pt2=POINTS[-1], 
                      color=(0,0,255), 
                      thickness=2)
        cv2.circle(img=image,
                   center=CENTER_OF_BOXES[-1], 
                   radius=6, 
                   color=(0,0,255), 
                   thickness=-1)
        cv2.putText(img=image,
                    text="{}".format(CENTER_OF_BOXES[-1]), 
                    org=CENTER_OF_BOXES[-1],  
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, 
                    color=(0,0,255), 
                    thickness=2)
        cv2.imshow(thisname, image)
        #cv2.waitKey(0)


def pixel_coordinates(d, args, which):
    """ 
    Use drag and drop to detect pixel coordinates. Save all points for one
    image, then repeat for the other camera image. 
    """
    global POINTS, CENTER_OF_BOXES, image
    if which == 'left':
        image = np.copy(d.left['raw'])
    elif which == 'right':
        image = np.copy(d.right['raw'])
    else:
        raise ValueError()
    num_points = 0
    cv2.destroyAllWindows()

    while num_points < args.num_per_img:
        window_name = "Currently have " +str(num_points)+ " points out of "+\
                str(args.num_per_img)+ " planned. Click and drag the NEXT "+\
                "direction NOW. Or press ESC to terminate the program."
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, click) # Record clicks to this window!
        cv2.resizeWindow(window_name, 1800, 2600)
        cv2.imshow(window_name, image)
        key = cv2.waitKey(0)
        if key in U.ESC_KEYS:
            sys.exit()
        num_points += 1
        cv2.destroyAllWindows()
        
    assert len(POINTS) == 2*len(CENTER_OF_BOXES) == 2*num_points == 2*args.num_per_img,\
            "{}, {}, {}, {}".format(len(POINTS), len(CENTER_OF_BOXES), num_points, args.num_per_img)
    info = {'points':POINTS, 'centers':CENTER_OF_BOXES}
    cv2.imwrite('calibration/calibrated_pts_' +which+ '.png', image)
    POINTS, CENTER_OF_BOXES = [], []
    return info


def calibrate(args, info_left, info_right, arm1, arm2, d):
    """ Perform camera calibration using both images and both arms.

    We have `info_left` and `info_right` which give us the 3D camera frame
    points for each of the (25 by default) grid points we're using. Iterate
    through these points, and for each, move PSM1 to it (putting end effector
    *inside* the grid) and then do the same for PSM2.
    """
    points_l = info_left['centers']
    points_r = info_right['centers']
    assert len(points_l) == len(points_r) == args.num_per_img
    num_points = args.num_per_img
    print("\nNow doing calibration with {} points".format(num_points))
    image = np.copy(d.left['raw'])

    # Save data to learn mapping.
    num_saved = 0
    map_info = defaultdict(list)

    for i,(pleft,pright) in enumerate(zip(points_l,points_r)):
        camera_pt = U.camera_pixels_to_camera_coords(pleft, pright)
        cX,cY = pleft

        # Deal with the image and get a visual.
        cv2.circle(image, (cX,cY), 50, (0,0,255))
        cv2.putText(img=image, text=str(num_saved), org=(cX,cY), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0,0,0), thickness=2)

        # Explicitly move LEFT arm to target and get information. Make sure to
        # move the right end-effector out of the way if needed.
        cv2.imshow("{} saved (out of {}). Move LEFT ARM in!!".format(num_saved,num_points), 
                   image)
        key = cv2.waitKey(0) 
        if key in U.ESC_KEYS:
            sys.exit()
        pos1, rot1 = U.pos_rot_arm(arm1, nparrays=True)

        # Pause to remind viewer to move the RIGHT arm, and get info.
        cv2.imshow("Move left out, and RIGHT ARM in!!".format(num_saved,num_points), image)
        key = cv2.waitKey(0) 
        if key in U.ESC_KEYS:
            sys.exit()
        pos2, rot2 = U.pos_rot_arm(arm2, nparrays=True)

        # Record information
        map_info['camera_pt'].append(camera_pt)
        map_info['left_arm_pos'].append(pos1)
        map_info['left_arm_rot'].append(rot1)
        map_info['right_arm_pos'].append(pos2)
        map_info['right_arm_rot'].append(rot2)
        num_saved += 1
        cv2.destroyAllWindows()
        print("\nFor point {}, we have:".format(i))
        print("\tcamera_pt: {}".format(camera_pt))
        print("\tleft pos:  {} and rot {}".format(pos1, rot1))
        print("\tright pos: {} and rot {}".format(pos2, rot2))
        arm1.close_gripper()
        arm2.close_gripper()

    return map_info


def solve_rigid_transform(A, B, debug=True):
    """
    Takes in two sets of corresponding points, returns the rigid transformation matrix 
    from the first to the second. This is a (3,4) matrix so we'd apply it on the original
    points in their homogeneous form with the fourth coordinate equal to one. This is
    slightly different from Brijen's code since I'm using np.array, not np.matrix.

    Notation: A for camera points, B for robot points, so want to find an affine mapping
    from A -> B with orthogonal rotation and a translation.
    """
    assert X.shape[0] == Y.shape[0] >= 3
    assert X.shape[1] == Y.shape[1] == 3
    A = X.T # (3,N)
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

    print("\nBegin debug prints for rigid transformation:")
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

    # Additional sanity checks. 
    if debug:
        print("\nCamera points (input), A.T:\n{}".format(A.T))
        print("Robot points (target), B.T:\n{}".format(B.T))
        print("Predicted robot points:\n{}".format(B_pred.T))
        print("Raw errors, B-B_pred:\n{}".format((B-B_pred).T))
        print("Mean abs error per dim: {}".format( (np.mean(np.abs(B-B_pred), axis=1))) )
        print("Residual (L2) for each:\n{}".format(l2_per_example.T))
        print("loss on data: {}".format(frobenius_loss))
        print("End of debug prints for rigid transformation.\n")
    return RB_matrix


def get_mapping(map_info):
    """ Whew, now finally we can get a mapping from camera to robot frames. 
    
    This must be done for each arm as they have different coordinate frames. For
    this we will use a rigid body transformation.
    """
    X = np.array(map_info['camera_pt'])
    Y_left  = np.array(map_info['left_arm_pos'])
    Y_right = np.array(map_info['right_arm_pos'])

    print("\nSome quick sanity checks on shapes for the rigid body transformation.")
    print("X.shape: {}".format(X.shape))
    print("Y_left.shape:  {}".format(Y_left.shape))
    print("Y_right.shape: {}".format(Y_right.shape))
    
    map_l = solve_rigid_transform(X, Y_left)
    map_r = solve_rigid_transform(X, Y_right)
    return map_l, map_r


if __name__ == "__main__":
    """ Can do everything, but comment out appropriately if pre-loading stuff. """
    pp = argparse.ArgumentParser() 
    pp.add_argument('--num_per_img', type=int, default=25)
    pp.add_argument('--dir_info_l', type=str, default='calibration/info_l_pts.p')
    pp.add_argument('--dir_info_r', type=str, default='calibration/info_r_pts.p')
    pp.add_argument('--dir_calib_data',  type=str, default='calibration/wrist_data.p')
    pp.add_argument('--dir_calib_map_l', type=str, default='calibration/wrist_map_l.p')
    pp.add_argument('--dir_calib_map_r', type=str, default='calibration/wrist_map_r.p')
    args = pp.parse_args()

    arm1, arm2, d = U.init(sleep_time=2)
    arm1.close_gripper()
    arm2.close_gripper()
    debug_images(d)

    # First, detect relevant points. Save if we want to continue from here.
    info_left  = pixel_coordinates(d, args, 'left') 
    info_right = pixel_coordinates(d, args, 'right')
    debug_info(info_left, info_right)
    U.store_pickle(args.dir_info_l, info=info_left)
    U.store_pickle(args.dir_info_r, info=info_right)

    # Second calibrate using both arms, in unison. If we load, uncomment two lines.
    #info_left  = U.load_pickle_to_list(args.dir_info_l, squeeze=True)
    #info_right = U.load_pickle_to_list(args.dir_info_r, squeeze=True)
    map_info = calibrate(args, info_left, info_right, arm1, arm2, d)
    U.store_pickle(args.dir_calib_data, info=map_info)

    # Third, get the mapping from `map_info`.
    #map_info = U.load_pickle_to_list(args.dir_calib_data, squeeze=True)
    map_l,map_r = get_mapping(map_info)
    U.store_pickle(args.dir_calib_map_l, info=map_l)
    U.store_pickle(args.dir_calib_map_r, info=map_r)
