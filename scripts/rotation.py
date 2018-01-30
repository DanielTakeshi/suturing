"""
A custom solution to help us get an analytic model for the rotations.  This code
answers: which order of (yaw, pitch, roll) is applied for rotations with the
dVRK.  Here's one way to do this (I'm not aware of a better way):

    0. Run this code a few times to get the dVRK at rotation (0, 0, 180) or
    another easy rotation to work with. Some manual tuning might be needed.

    1. With SNAP, have the dVRK grip a needle. The needle's tip will be used for
    determining positions. It doesn't have to be painted as we'll be clicking
    its tip. At this point, I need to compute the position of the needle wrt the
    tool frame!

    2. With a known fixed position vector, we set the orientation of the dVRK to
    have zero value for yaw, pitch, and roll. Hence the rotation R_{st} matrix
    is the identity.  UPDATE: while this is ideal, we cannot get zero roll by
    design of the dVRK.  The ranges are as follows:

        - YAW:   [-180, 180] # I think this has the full 360 degrees range.
        - PITCH: [-50, 50] # Limited, moving wrist (w/palm facing down) up/down.
        - roll:  [-180, -100] U [100, 180] # Limited, moving a wrist sideways.

        And while we can get yaw and pitch as zero, the roll seems easiest to
        set at 180 (or equivalently, -180, moving clockwise).  Fortunately this
        should only be a straight-up rotation matrix about the z-axis. Well, I
        *hope* it's the z-axis ... but as long as we are consistent does it
        matter? As in, I'm consistent with assigning the third element of the
        rotation vector as the roll?

    3. Determine the p_t of the needle's tip wrt the TOOL frame. This is a
    matter of subtracting vectors and using that R_z rotation matrix.
    
    4. Now we can command the dVRK to go to different, known rotations.
    Fortunately, p_t is STILL THE SAME, and p_s will be computed via the user
    clicking on the image to indicate the position. (And then using the
    camera-to-base mapping.)

    5. With our set of p_s = t_{st} + R_{st}p_t equations, we simply solve for
    R_{st} and see which of the known, fixed orderings of yaw,pitch,roll matches
    closest to the matrix wrt Frobenius norm.

This approach depends on the calibration being done correctly, with the grid of
'square ridges' in it. That's not actually something we can rely on. It also
assumes that the dVRK is 'accurate enough' when it gets rotated, which again is
not guaranteed.

(c) 2018 by Daniel Seita
"""
from autolab.data_collector import DataCollector
from collections import defaultdict
from dvrk import robot
import argparse, cv2, os, pickle, sys, tfx, time
import numpy as np
import utils as U
np.set_printoptions(suppress=True, precision=5)

# ADJUST!
PATH_CALIB = 'calibration/'
ROT_FILE = 'scripts/files_rotations/rotations_data.p'
TIP_FILE = 'scripts/files_rotations/needle_tip_data.p'
Z_OFFSET = 0.008
HOME_POS_ARM1 = [0.0982,  0.0126, -0.105]
HOME_ROT_ARM1 = [0.0,     0.0,     180.0]


def get_in_good_starting_position(arm, which='arm1'):
    """ 
    Only meant so we get a good starting position, to compensate for how
    commanding the dVRK to go to a position/rotation doesn't actually work that
    well, particularly for the rotations. Some human direct touch is needed.
    """
    assert which == 'arm1'
    pos, rot = U.pos_rot_arm(arm, nparrays=True)
    print("(starting method) starting position and rotation:")
    print(pos, rot)
    U.move(arm, HOME_POS_ARM1, HOME_ROT_ARM1, speed='slow')
    time.sleep(2)
    print("(starting method) position and rotation after moving:")
    pos, rot = U.pos_rot_arm(arm, nparrays=True)
    print(pos, rot)
    print("(Goal was: {} and {}".format(HOME_POS_ARM1, HOME_ROT_ARM1))
    R = U.rotation_matrix_3x3_axis(angle=180, axis='z')
    print("With desired rotation matrix:\n{}".format(R))
    print("Now exiting...")
    sys.exit()


# For the mouse callback method, dragging boxes on images.
POINTS = []
CENTERS = []
image = None


def click(event, x, y, flags, param):
    global POINTS, CENTERS, image
             
    # If left mouse button clicked, record the starting (x,y) coordinates.
    if event == cv2.EVENT_LBUTTONDOWN:
        POINTS.append( (x,y) )
                                                 
    # Check to see if the left mouse button was released.
    elif event == cv2.EVENT_LBUTTONUP:
        POINTS.append( (x,y) )
        upper_left  = POINTS[-2]
        lower_right = POINTS[-1]
        assert len(POINTS) % 2 == 0
        assert upper_left[0] < lower_right[0]
        assert upper_left[1] < lower_right[1]
        center_x = int(upper_left[0] + (lower_right[0]-upper_left[0])/2)
        center_y = int(upper_left[1] + (lower_right[1]-upper_left[1])/2)
        CENTERS.append( (center_x,center_y) )
        cv2.destroyAllWindows()

        thisname = "Inside `click` AFTER dragging a rectangle. DON'T CLICK: press a key!"
        cv2.namedWindow(thisname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(thisname, 1800, 2600)
        cv2.rectangle(img=image,
                      pt1=upper_left,
                      pt2=lower_right,
                      color=(255,0,0), 
                      thickness=2)
        cv2.circle(img=image,
                   center=CENTERS[-1], 
                   radius=6, 
                   color=(0,0,255), 
                   thickness=-1)
        cv2.putText(img=image,
                    text="{}".format(CENTERS[-1]), 
                    org=CENTERS[-1],  
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, 
                    color=(0,0,255), 
                    thickness=2)
        cv2.imshow(thisname, image)


def get_tip_needle(arm, d, idx, rot_targ):
    """ 
    Now the user has to click to get the needle tips. We click twice, one for
    the left and one for the right, and return the camera coordinates.
    """
    global CENTERS, image
    pos, rot = U.pos_rot_arm(arm, nparrays=True)
    old_len = len(CENTERS)

    # Left camera image.
    w = "{}-th rot_targ: {}, left camera. Drag boxes on needle tip, then SPACE.".format(
            idx, rot_targ)
    image = np.copy(d.left['raw'])
    cv2.namedWindow(w, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(w, click)
    cv2.resizeWindow(w, 1800, 2600)
    cv2.imshow(w, image)
    key = cv2.waitKey(0)
    if key in U.ESC_KEYS:
        sys.exit()
    cv2.destroyAllWindows()

    # Right camera image.
    w = "{}-th rot_targ: {}, right camera. Drag boxes on needle tip, then SPACE.".format(
            idx, rot_targ)
    image = np.copy(d.right['raw'])
    cv2.namedWindow(w, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(w, click)
    cv2.resizeWindow(w, 1800, 2600)
    cv2.imshow(w, image)
    key = cv2.waitKey(0)
    if key in U.ESC_KEYS:
        sys.exit()
    cv2.destroyAllWindows()   

    assert len(CENTERS) == old_len + 2
    assert len(POINTS) % 2 == 0
    needle_tip_c = U.camera_pixels_to_camera_coords(CENTERS[-2], CENTERS[-1], nparrays=True)
    return needle_tip_c


def collect_tip_data(arm, R_real, R_desired, wrist_map_c2l, d):
    """ Collects data points on the needle tips.
    
    We want to be at rotation [0, 0, 180] (equivalently, -180 for the last part,
    the `roll`) so that we can assume we know the rotation matrix. Assumes that
    the roll belongs to the z-axis. Due to imperfections, we won't be exact.

    The goal here is to determine as best an estimate of the position of the
    needle TIP wrt the TOOL frame as possible. We'll run this through several
    different positions, each time clicking to get an estimate, and then we
    average those together.
    """
    data = defaultdict(list)
    home_pos = np.copy(HOME_POS_ARM1)
    home_rot = np.copy(HOME_ROT_ARM1)
    assert (home_rot[0] == 0 and home_rot[1] == 0 and home_rot[2] == 180)
    idx = 0

    # Slightly adjust position to get different data points in x-y dims.
    deltas_meters = [-0.009, -0.003, 0.003, 0.009]

    # NOW begin the loop over different possible positions. Obviously we need to
    # always go to the `home_rot`, though some imprecisions will result.
    for dx in deltas_meters:
        for dy in deltas_meters:
            idx += 1
            pos_tool_wrt_base_targ = [home_pos[0]+dx, home_pos[1]+dy, home_pos[2]]
            U.move(arm, pos_tool_wrt_base_targ, home_rot)
            time.sleep(2)
            pos_tool_wrt_base_code, rot_tool_wrt_base_code = U.pos_rot_arm(arm, nparrays=True)

            # A human now clicks on the windows to get needle TIP position.
            # Unfortunately this calibration may be quite erroneous, but if we
            # save all the files we might be able to re-run w/out re-clicking.
            needle_tip_c = get_tip_needle(arm, d, idx, home_rot)
            needle_tip_c_h = np.concatenate( (needle_tip_c,np.ones(1)) )
            needle_tip_l = wrist_map_c2l.dot(needle_tip_c_h)

            # Compute the needle tip wrt TOOL frame; approximate with R_z(roll).
            roll = rot_tool_wrt_base_code[2]
            R = U.rotation_matrix_3x3_axis(angle=roll, axis='z')
            Rinv = np.linalg.inv(R)
            needle_tip_r = Rinv.dot( needle_tip_l - pos_tool_wrt_base_code )

            # Bells and whistles, a bit inefficient but whatever.
            data['pos_tool_wrt_s_targ'].append(pos_tool_wrt_base_targ)
            data['pos_tool_wrt_s_code'].append(pos_tool_wrt_base_code)
            data['rot_tool_wrt_s_targ'].append(home_rot)
            data['rot_tool_wrt_s_code'].append(rot_tool_wrt_base_code)
            data['pos_ntip wrt_c'].append(needle_tip_c)
            data['pos_ntip_wrt_s'].append(needle_tip_l)
            data['pos_ntip_wrt_r'].append(needle_tip_r)

            print("\nAdding {}-th data point for `collect_tip_data()`.".format(idx))
            print("pos_tool_wrt_s_targ:  {}".format(np.array(pos_tool_wrt_base_targ)))
            print("pos_tool_wrt_s_code:  {}".format(pos_tool_wrt_base_code))
            print("rot_tool_wrt_s_targ:  {}".format(home_rot))
            print("rot_tool_wrt_s_code:  {}".format(rot_tool_wrt_base_code))
            print("pos_ntip wrt_c:       {}".format(needle_tip_c))
            print("pos_ntip_wrt_s:       {}".format(needle_tip_l))
            print("pos_ntip_wrt_r:       {}".format(needle_tip_r))
                
    # This was the whole point of the method.
    print("\nFinished collecting data for computing needle_tip_r.")
    n_t = np.zeros(3)
    for idx,item in enumerate(data['pos_ntip_wrt_r']):
        print("{}, {}".format(item, idx))
        n_t += item
    n_t /= len(data['pos_ntip_wrt_r'])
    print("average `n_t` to use later: {}\n".format(n_t))
    return data


def collect_data(arm, R, wrist_map_c2l, d):
    """ Collects data points to determine which rotation matrix we're using.

    The `t_st` and `R_st` define the rigid body from the tool to the arm base.
    Don't forget that the needle must be gripped with SNAP so that the tip
    vector `n_t` remains consistent wrt the tool (t) frame.

    Needs to be called directly after `collect_tip_data()` so that the position
    of the needle tip wrt the TOOL frame is the same as earlier when we were
    explicitly computing/estimating that in `collect_tip_data()`. We don't
    actually need the `n_t` data here as that's needed in the LAST step.
    """
    R_z = R
    pos, rot = U.pos_rot_arm(arm, nparrays=True)
    t_st = pos
    data = defaultdict(list)

    # We'll test out these equally-spaced values. Note 180 = -180.
    yaws    = [-30, 0, 30]
    pitches = [-30, 0, 30]
    rolls   = [-160, -180, 160]
    idx = 0

    # --------------------------------------------------------------------------
    # I think it helps the dVRK to move to the first rotation *incrementally*.
    print("We first incrementally move to the starting rotation ...")

    U.move(arm, pos=t_st, rot=[0, 0, -160])
    time.sleep(1)
    pos, rot = U.pos_rot_arm(arm, nparrays=True)
    print("we have pos, rot: {}, {}".format(pos, rot))

    U.move(arm, pos=t_st, rot=[0, -30, -160])
    time.sleep(1)
    pos, rot = U.pos_rot_arm(arm, nparrays=True)
    print("we have pos, rot: {}, {}".format(pos, rot))

    U.move(arm, pos=t_st, rot=[-30, -30, -160])
    time.sleep(1)
    pos, rot = U.pos_rot_arm(arm, nparrays=True)
    print("we have pos, rot: {}, {}".format(pos, rot))

    print("(End of incrementally moving to start)\n")
    # --------------------------------------------------------------------------

    # NOW begin the loop over different possible rotations.
    for alpha in yaws:
        for beta in pitches:
            for gamma in rolls:
                idx += 1
                rot_targ = [alpha, beta, gamma]
                U.move(arm, pos=t_st, rot=rot_targ)
                time.sleep(2)
                pos, rot = U.pos_rot_arm(arm, nparrays=True)

                # A human now clicks on the windows to get needle TIP position.
                # Actually we probably shouldn't use this specific map as we may
                # get better calibration later, but it doesn't hurt to include.
                needle_tip_c = get_tip_needle(arm, d, idx, rot_targ)
                needle_tip_c_h = np.concatenate( (needle_tip_c,np.ones(1)) )
                needle_tip_l = wrist_map_c2l.dot(needle_tip_c_h)

                # Bells and whistles, a bit inefficient but whatever.
                data['pos_tool_wrt_s_targ'].append(t_st)
                data['pos_tool_wrt_s_code'].append(pos)
                data['rot_tool_wrt_s_targ'].append(rot_targ)
                data['rot_tool_wrt_s_code'].append(rot)
                data['pos_ntip_wrt_c_clicks'].append(needle_tip_c)
                data['pos_ntip_wrt_s'].append(needle_tip_l)

                print("\nAdding {}-th data point".format(idx))
                print("TARGET (yaw,pitch,roll):  {}".format(rot_targ))
                print("Actual (yaw,pitch,roll):  {}".format(rot))
                print("Target pos (i.e., t_st):  {}".format(t_st))
                print("Actual pos from command:  {}".format(pos))

    return data


def determine_rotation(arm, d, tip_data, rot_data):
    """ Determines the rotation matrix based on data from `collect_data()`.  
    
    We use the same `n_t` computed from earlier for all data points, and
    evaluate based on L2 norm. For the translation vector and the rotations, we
    use what is derived form code, which is slightly different from the value we
    commanded the dVRK to go to (obviously).
    """
    n_t = np.zeros(3)
    for this_n_t in tip_data['pos_ntip_wrt_r']:
        n_t += this_n_t
    n_t /= len(tip_data['pos_ntip_wrt_r'])
    print("Our n_t to use in this stage: {}".format(n_t))

    K = len(rot_data['pos_ntip_wrt_s'])
    errors_zyz = []
    errors_zyx = []

    for k in range(K):
        lhs  = rot_data['pos_ntip_wrt_s'][k]
        t_st = rot_data['pos_tool_wrt_s_code'][k]
        ypr  = rot_data['rot_tool_wrt_s_code'][k]
        yaw, pitch, roll = ypr[0], ypr[1], ypr[2]

        # R_zyz
        R_z1 = U.rotation_matrix_3x3_axis(angle=roll,  axis='z')
        R_y  = U.rotation_matrix_3x3_axis(angle=pitch, axis='y')
        R_z2 = U.rotation_matrix_3x3_axis(angle=yaw,   axis='z')
        R_zyz = R_z2.dot(R_y).dot(R_z1)

        # R_zyx
        R_x = U.rotation_matrix_3x3_axis(angle=roll,  axis='x')
        R_y = U.rotation_matrix_3x3_axis(angle=pitch, axis='y')
        R_z = U.rotation_matrix_3x3_axis(angle=yaw,   axis='z')
        R_zyx = R_z.dot(R_y).dot(R_x)

        # Evaluate!
        rhs_zyz = t_st + R_zyz.dot( n_t )
        rhs_zyx = t_st + R_zyx.dot( n_t )
        err_zyz = np.linalg.norm(lhs - rhs_zyz)
        err_zyx = np.linalg.norm(lhs - rhs_zyx)
        errors_zyz.append( err_zyz )
        errors_zyx.append( err_zyx )
        print("\nerr_zyz: {:.3f} for {}-th sample".format(err_zyz, k))
        print("err_zyx: {:.3f} for {}-th sample".format(err_zyx, k))
        print("R_zyz:\n{}".format(R_zyz))
        print("R_zyx:\n{}".format(R_zyx))

    print("\nDone with evaluation!")
    print("zyz has avg error {:.5f}".format(np.mean(errors_zyz)))
    print("zyx has avg error {:.5f}".format(np.mean(errors_zyx)))


if __name__ == "__main__":
    arm1, _, d = U.init()
    wrist_map_c2l = U.load_pickle_to_list(PATH_CALIB+'wrist_map_c2l.p', squeeze=True)
    wrist_map_c2r = U.load_pickle_to_list(PATH_CALIB+'wrist_map_c2r.p', squeeze=True)
    wrist_map_l2r = U.load_pickle_to_list(PATH_CALIB+'wrist_map_l2r.p', squeeze=True)

    # We're on stage 1, 2, or 3. ***ADJUST THIS***.
    stage = 3

    if stage == 1:
        get_in_good_starting_position(arm1)

    elif stage == 2:
        assert not os.path.isfile(ROT_FILE)
        assert not os.path.isfile(TIP_FILE)
        arm1.open_gripper(60) 
        time.sleep(2) # PUT NEEDLE INSIDE THE GRIPPER, WITH SNAP!
        arm1.close_gripper()
        pos, rot = U.pos_rot_arm(arm1, nparrays=True)
        print("starting position and rotation:")
        print(pos, rot)
        print("HOME_POS_ARM1: {}".format(HOME_POS_ARM1))
        print("HOME_ROT_ARM1: {}".format(HOME_ROT_ARM1))
        R_real    = U.rotation_matrix_3x3_axis(angle=rot[2], axis='z')
        R_desired = U.rotation_matrix_3x3_axis(angle=180, axis='z')
        print("With actual rotation matrix:\n{}".format(R_real))
        print("With desired rotation matrix:\n{}".format(R_desired))

        # Get position of the needle TIP wrt the TOOL frame.
        tip_data = collect_tip_data(arm1, R_real, R_desired, wrist_map_c2l, d)
        U.store_pickle(fname=TIP_FILE, info=tip_data)

        # Now collect data from different rotations, using SAME needle grip.
        # This is why the code must be run sequentially in the same stage.
        rot_data = collect_data(arm1, R_real, wrist_map_c2l, d)
        U.store_pickle(fname=ROT_FILE, info=rot_data)

    elif stage == 3:
        # Collect data and determine most accurate rotation matrix.
        tip_data = U.load_pickle_to_list(TIP_FILE, squeeze=True)
        rot_data = U.load_pickle_to_list(ROT_FILE, squeeze=True)
        determine_rotation(arm1, d, tip_data, rot_data)
    else: 
        raise ValueError()
