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
import argparse, cv2, pickle, os, sys, time
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
        cv2.imshow("{} saved (out of {}). LEFT ARM in, press SPACE".format(num_saved,num_points), 
                   image)
        key = cv2.waitKey(0) 
        if key in U.ESC_KEYS:
            sys.exit()
        pos1, rot1 = U.pos_rot_arm(arm1, nparrays=True)
        time.sleep(1)

        # Pause to remind viewer to move the RIGHT arm, and get info.
        cv2.imshow("Put left to next, RIGHT ARM in!! (press SPACE)".format(num_saved,num_points), 
                   image)
        key = cv2.waitKey(0) 
        if key in U.ESC_KEYS:
            sys.exit()
        pos2, rot2 = U.pos_rot_arm(arm2, nparrays=True)
        time.sleep(1)

        # Record information
        map_info['camera_pt'].append(camera_pt)
        map_info['left_arm_pos'].append(pos1)
        map_info['left_arm_rot'].append(rot1)
        map_info['right_arm_pos'].append(pos2)
        map_info['right_arm_rot'].append(rot2)
        num_saved += 1
        cv2.destroyAllWindows()
        print("\nFor point {}, we have:".format(i))
        print("\tcamera_pt: {}".format(np.array(camera_pt)))
        print("\tleft pos:  {} and rot {}".format(pos1, rot1))
        print("\tright pos: {} and rot {}".format(pos2, rot2))
        arm1.close_gripper()
        arm2.close_gripper()

    return map_info


def get_mapping(map_info):
    """ Get a mapping from camera to robot frames. 
    
    The two arms have different base frames. We also get a map from the left
    base frame to the right base frame. Actually we only need two of these but
    I guess three doesn't hurt.
    """
    X_camera = np.array(map_info['camera_pt'])
    Y_left   = np.array(map_info['left_arm_pos'])
    Y_right  = np.array(map_info['right_arm_pos'])

    print("\nQuick sanity checks on shapes for the rigid body transformation.")
    print("X_camera.shape: {}".format(X_camera.shape))
    print("Y_left.shape:   {}".format(Y_left.shape))
    print("Y_right.shape:  {}".format(Y_right.shape))
    
    map_c2l = U.solve_rigid_transform(X_camera, Y_left)
    map_c2r = U.solve_rigid_transform(X_camera, Y_right)
    map_l2r = U.solve_rigid_transform(Y_left,   Y_right)
    return (map_c2l, map_c2r, map_l2r)


if __name__ == "__main__":
    """ Can do everything, but comment out appropriately if pre-loading stuff. """
    h = 'calibration/'
    pp = argparse.ArgumentParser() 
    pp.add_argument('--num_per_img',    type=int, default=25)
    pp.add_argument('--dir_info_left',  type=str, default=h+'wrist_l_pts.p')
    pp.add_argument('--dir_info_right', type=str, default=h+'wrist_r_pts.p')
    pp.add_argument('--dir_calib_data', type=str, default=h+'wrist_data.p')
    pp.add_argument('--dir_calib_c2l',  type=str, default=h+'wrist_map_c2l.p')
    pp.add_argument('--dir_calib_c2r',  type=str, default=h+'wrist_map_c2r.p')
    pp.add_argument('--dir_calib_l2r',  type=str, default=h+'wrist_map_l2r.p')
    args = pp.parse_args()

    arm1, arm2, d = U.init(sleep_time=2)
    arm1.close_gripper()
    arm2.close_gripper()
    debug_images(d)

    # First, detect relevant points. Save if we want to continue from here.
    info_left  = pixel_coordinates(d, args, 'left') 
    info_right = pixel_coordinates(d, args, 'right')
    debug_info(info_left, info_right)
    U.store_pickle(args.dir_info_left,  info=info_left)
    U.store_pickle(args.dir_info_right, info=info_right)

    # Second calibrate using both arms, in unison. If we load, uncomment two lines.
    #info_left  = U.load_pickle_to_list(args.dir_info_left,  squeeze=True)
    #info_right = U.load_pickle_to_list(args.dir_info_right, squeeze=True)
    map_info = calibrate(args, info_left, info_right, arm1, arm2, d)
    U.store_pickle(args.dir_calib_data, info=map_info)

    # Third and lastly, get the mapping from `map_info`.
    #map_info = U.load_pickle_to_list(args.dir_calib_data, squeeze=True)
    (map_c2l, map_c2r, map_l2r) = get_mapping(map_info)
    U.store_pickle(args.dir_calib_c2l, info=map_c2l)
    U.store_pickle(args.dir_calib_c2r, info=map_c2r)
    U.store_pickle(args.dir_calib_l2r, info=map_l2r)
