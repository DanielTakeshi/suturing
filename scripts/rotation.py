"""
A custom solution to help us get an analytic model for the rotations.  This code
answers: which order of (yaw, pitch, roll) is applied for rotations with the
dVRK.  Here's one way to do this (I'm not aware of a better way):

    0. Run this code a few times to get the dVRK at rotation (0, 0, 180) or
    another easy rotation to work with. Some manual tuning might be needed.

    1. With SNAP, have the dVRK grip a needle. The needle's tip will be used for
    determining positions. It doesn't have to be painted as we'll be clicking
    its tip.

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
Z_OFFSET = 0.008
HOME_POS_ARM1 = [0.0982,  0.0226, -0.105]
HOME_ROT_ARM1 = [0.0,     0.0,     160.0]


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
    print("With rotation matrix:\n{}".format(R))
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


def click_stuff(which):
    """ 
    Putting this here since we repeat for left and right camera images.  We
    actually use this for both offset computation AND the evaluation stage, but
    we only run one of those in the code which means CENTERS and POINTS won't
    mess us up with unexpected values.
    """
    global CENTERS, image
    if which == 'left':
        image = np.copy(d.left['raw'])
    elif which == 'right':
        image = np.copy(d.right['raw'])
    max_points = 2
    current_points = 0
    old_len = len(CENTERS)

    while current_points < max_points:
        name = "{} image w/{} current pts so far. DRAG BOXES around tip, THEN space bar, THEN gripper, THEN space bar".format(which.upper(), current_points)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(name, click) # Record clicks to this window!
        cv2.resizeWindow(name, 1800, 2600)
        cv2.imshow(name, image)
        key = cv2.waitKey(0)
        if key in U.ESC_KEYS:
            sys.exit()
        cv2.destroyAllWindows()
        current_points += 1

    assert len(CENTERS) == old_len + 2
    assert len(POINTS) % 2 == 0
    print("finished clicking on {} image".format(which))


def offsets(arm, d, args, wrist_map_c2l, wrist_map_c2r, wrist_map_l2r):
    """ Stage 1. 
    
    1. Before running the python script, get the setup as shown in the image in
    the README where the needle is on the foam and the dVRK (w/SNAP) is about to
    grip it. 
    
    2. Then, this code waits until the user has manipulated master tools to pick
    up the needle at some location. EDIT: actually it's easier to just move and
    then assume we translate up by a few mm. I'm getting lack of responses with
    the master tools, for some reason. So just move it directly and explicitly.
    
    3. Images will pop-up. Press ESC to terminate the program. Otherwise, we
    click and crop: (a) the end-point of the needle AND (b) the place where the
    dVRK grasps the needle. The latter is mainly to see if this coincides with
    the dVRK's actual position. MUST BE DONE IN ORDER!
    
    4. This saves into a pickle file by appending, so in theory we continually
    add data points. (The main catch is to check that if we change the setup,
    we're NOT counting those older points.)
    
    Repeat the process by moving the master tools again, and go to step 2.
    """
    global CENTERS, image
    which = 'left' if args.arm==1 else 'right'
    pname = args.path_offsets+'/offsets_{}_arm_data.p'.format(which)
    if which == 'left':
        R = wrist_map_c2l
    else:
        R = wrist_map_c2r
    num_offsets = 0

    while True:
        # Just an overall view before grabbing needle. OK to only see the left
        # image. And BTW, this is when we change the dVRK to get more data.
        arm.open_gripper(50)
        print("\nNow starting new offset with {} offsets so far...".format(num_offsets))
        name = "Left image w/{} offsets so far. MOVE DVRK, press any key".format(num_offsets)
        U.call_wait_key( cv2.imshow(name, d.left['raw']) )
        arm.close_gripper()
        pos,rot = U.pos_rot_arm(arm, nparrays=True)
        pos[2] += args.z_height
        U.move(arm, pos, rot)
        time.sleep(2)

        # The user now clicks stuff. LEFT CAMERA, then RIGHT CAMERA.
        click_stuff('left')
        click_stuff('right')

        # Compute data to store. We stored left tip, left grip, right tip, right grip.
        pos_g,rot_g = U.pos_rot_arm(arm, nparrays=True)
        assert len(CENTERS) % 4 == 0, "Error, len(CENTERS): {}".format(len(CENTERS))
        camera_tip  = U.camera_pixels_to_camera_coords(CENTERS[-4], CENTERS[-2], nparrays=True)
        camera_grip = U.camera_pixels_to_camera_coords(CENTERS[-3], CENTERS[-1], nparrays=True)

        # Map stuff to stuff. The `h` in `xyz_h` refers to homogeneous coordinates.
        ct_h = np.concatenate( (camera_tip,np.ones(1)) )
        cg_h = np.concatenate( (camera_grip,np.ones(1)) )
        base_t = R.dot(ct_h)
        base_g = R.dot(cg_h)
        camera_dist = np.linalg.norm( camera_tip-camera_grip )

        # Distance based on click interace (should be same as camera) and based
        # on dvrk, using cartesian position function (this will be different).
        # Unfortunately the dvrk one is the easiest to use in real applications.
        base_dist_click = np.linalg.norm( base_t-base_g )
        base_dist_dvrk  = np.linalg.norm( base_t-pos_g )
        assert np.abs(base_dist_click - camera_dist) < 1e-5

        # Compute offset vector wrt base frame. I don't think the order of
        # subtraction matters as long as in real applications, we are consistent
        # with usage, so in application we'd do (pos_n_tip_wrt_base)-(offset).
        offset_wrt_base_click = base_t - base_g
        offset_wrt_base_dvrk  = base_t - pos_g

        phi_c, psi_c = compute_phi_and_psi(d=base_dist_click)
        phi_d, psi_d = compute_phi_and_psi(d=base_dist_dvrk)


def evaluate(arm, d, args, wrist_map_c2l, wrist_map_c2r, wrist_map_l2r):
    """ Stage 2. 
    
    Testing stage. At a high level, move the dVRK end-effector to various
    locations and once again explicitly record needle tips. Steps:

    1. Start with the dVRK gripping a needle. Try different orientations, to
    ensure that performance is orientation-invariant. This time, we should have
    the suturing phantom there! For now we still have the click interface to
    accurately compute psi.

    2. To get psi, images pop up from left and right cameras. Press ESC to
    terminate the program. If we want to proceed, then click and drag boxes of
    target locations for the needle tip.
    
    3. This time, we let the needle move. We do NOT explicitly compute an offset
    vector because we do not know where the dVRK's end-effector goes.
    """
    global CENTERS, image
    which = 'left' if args.arm==1 else 'right'
    pname = args.path_offsets+'/offsets_{}_arm_data.p'.format(which)
    if which == 'left':
        R = wrist_map_c2l
    else:
        R = wrist_map_c2r
    offsets = U.load_pickle_to_list(pname)
    print("loaded offsets, length: {}".format(len(offsets)))

    # Do stuff here just to grip the needle and get set up for computing phi/psi.
    arm.open_gripper(50)
    name = "Left image for evaluation stage. MOVE DVRK, then press any key"
    U.call_wait_key( cv2.imshow(name, d.left['raw']) )
    arm.close_gripper()
    pos,rot = U.pos_rot_arm(arm, nparrays=True)
    pos[2] += args.z_height
    U.move(arm, pos, rot)
    time.sleep(2)

    # The user now clicks stuff. LEFT CAMERA, then RIGHT CAMERA.
    click_stuff('left')
    click_stuff('right')
    assert len(CENTERS) == 4, "Error, len(CENTERS): {}".format(len(CENTERS))
    assert len(POINTS) == 8, "Error, len(POINTS): {}".format(len(POINTS))

    # New to this method: move the dVRK to a different location.
    print("Not moving to a new spot, for now.")

    # Now we can actually get phi/psi.
    pos_g,rot_g = U.pos_rot_arm(arm, nparrays=True)
    camera_tip  = U.camera_pixels_to_camera_coords(CENTERS[-4], CENTERS[-2], nparrays=True)
    camera_grip = U.camera_pixels_to_camera_coords(CENTERS[-3], CENTERS[-1], nparrays=True)
    ct_h        = np.concatenate( (camera_tip,np.ones(1)) )
    cg_h        = np.concatenate( (camera_grip,np.ones(1)) )
    base_t      = R.dot(ct_h)
    base_g      = R.dot(cg_h)
    camera_dist = np.linalg.norm( camera_tip-camera_grip )

    # Distance based on click interace (should be same as camera) and based
    # on dvrk, using cartesian position function (this will be different).
    # Unfortunately the dvrk one is the easiest to use in real applications.
    base_dist_click = np.linalg.norm( base_t-base_g )
    base_dist_dvrk  = np.linalg.norm( base_t-pos_g )
    assert np.abs(base_dist_click - camera_dist) < 1e-5

    # Compute phi and psi based on the STARTING configuration.
    phi_c, psi_c = compute_phi_and_psi(d=base_dist_click)
    phi_d, psi_d = compute_phi_and_psi(d=base_dist_dvrk)

    # --------------------------------------------------------------------------
    # Compute offset vector wrt base. I don't think the order of subtraction
    # matters as long as in real applications, we are consistent with usage, so
    # in application we'd do (pos_n_tip_wrt_base)-(offset). 
    #
    # NOTE: this is only the offset vector wrt the starting position that we
    # have, but there are two hypotheses. 
    #
    #   1. That we can avoid computing these if we pre-compute before hand (as
    #   we have code for) so maybe we don't need a click interface. 
    #   2. To get the needle to go somewhere, it's a matter of doing the
    #   arithmetic as specified earlier in the comments! 
    #
    # For now we'll compute the two offsets wrt base because we might as well
    # use it to simplify the task (but this is only simplifying the detection of
    # the needle tip and where the dVRK grips it).
    # --------------------------------------------------------------------------
    offset_wrt_base_click = base_t - base_g
    offset_wrt_base_dvrk  = base_t - pos_g
    offset_saved =  get_offset(offsets, psi=psi_c, kind='click', which=which, debug=True)

    print("offset_wrt_base_click: {}".format(offset_wrt_base_click))
    print("offset_wrt_base_dvrk:  {}".format(offset_wrt_base_dvrk))
    print("offset_saved:          {}".format(offset_saved))


def collect_data(arm, R, wrist_map_c2l):
    """ Collects data points to determine which rotation matrix we're using.

    The `t_st` and `R_st` define the rigid body from the tool to the arm base.
    Don't forget that the needle must be gripped with SNAP so that the tip
    vector `n_t` remains consistent wrt the tool (t) frame.
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
                wname = "our {}-th TARGET rotation is {}".format(idx, rot_targ)
                # TODO

                # Bells and whistles, a bit inefficient but whatever.
                data['pos_s_tip'].append(None) # TODO
                data['pos_t_tip'].append(None) # TODO
                data['t_st'].append(t_st)
                data['pos_s_tool'].append(pos)
                data['rot_s_targ'].append(rot_targ)
                data['rot_s_real'].append(rot)
                print("\nAdding {}-th data point".format(idx))
                print("TARGET (yaw,pitch,roll):  {}".format(rot_targ))
                print("Actual (yaw,pitch,roll):  {}".format(rot))
                print("Target pos (i.e., t_st):  {}".format(t_st))
                print("Actual pos from command:  {}".format(pos))

    return data


def determine_rotation(data):
    """ Determines the rotation matrix based on data from `collect_data()`. """
    pass


if __name__ == "__main__":
    assert not os.path.isfile(ROT_FILE)
    arm1, _, d = U.init()
    wrist_map_c2l = U.load_pickle_to_list(PATH_CALIB+'wrist_map_c2l.p', squeeze=True)
    wrist_map_c2r = U.load_pickle_to_list(PATH_CALIB+'wrist_map_c2r.p', squeeze=True)
    wrist_map_l2r = U.load_pickle_to_list(PATH_CALIB+'wrist_map_l2r.p', squeeze=True)

    # If necessary. Comment this out as needed.
    #get_in_good_starting_position(arm1)

    # Otherwise we proceed with what we really want: collecting data.
    arm1.open_gripper(60)
    time.sleep(2)
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

    # Collect data and determine most accurate rotation matrix.
    data = collect_data(arm1, R_real, wrist_map_c2l)
    U.store_pickle(fname=ROT_FILE, info=data)
    determine_rotation(data)
