"""
Follow my notes and get an analytic model of the needle. I'm trying to control
the robot so that the needle tip ends up in the correct location.  This should
be a self-contained script purely for learning and testing the analytic model.

Call code in stages, so `python analytic_model.py --stage X --arm Y` where X==1
is the stage when we compute offsets, and X==2 is when we evaluate. Also, the
reason why we do separate arms at a time is that the two arms have different
base frames, so it's a bit easier to separate them. See the methods for more
detailed documentation. Also, it's mainly the left arm that we care about.

(c) 2018 by Daniel Seita
"""
from autolab.data_collector import DataCollector
from collections import defaultdict
from dvrk import robot
import argparse, cv2, os, pickle, sys, tfx, time
import numpy as np
import utils as U
np.set_printoptions(suppress=True, precision=5)

# For the mouse callback method, dragging boxes on images.
POINTS = []
CENTERS = []
image = None


def debug_images(d, args):
    """ Keep this here in case I need to debug images. """
    cv2.imshow("Left Camera Image (debugging only, DON'T CLICK!)", d.left['raw'])
    cv2.waitKey(0)
    cv2.imshow("Right Camera Image (debugging only, DON'T CLICK!)", d.right['raw'])
    cv2.waitKey(0)
    cv2.imwrite(args.path_offsets+'/raw_left_arm_'+str(args.arm)+'.png', d.left['raw'])
    cv2.imwrite(args.path_offsets+'/raw_right_arm_'+str(args.arm)+'.png', d.right['raw'])


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
    """ Putting this here since we repeat for left and right camera images. """
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
    the master tools, for some reason.
    
    3. Images will pop-up. Press ESC to terminate the program. Otherwise, we
    click and crop: (a) the end-point of the needle AND (b) the place where the
    dVRK grasps the needle. The latter is mainly to see if this coincides with
    the dVRK's actual position. MUST BE DONE IN ORDER!
    
    4. This saves into a pickle file by appending, so in theory we just
    continually add data points. (The main catch is to check that if we change
    the setup, we're NOT counting those older points ... we'll just have to move
    them to separate files or delete them.) 
    
    Repeat the process by moving the master tools again, and go to step 2.
    """
    global CENTERS, image
    which = 'left' if args.arm==1 else 'right'
    pname = args.path_offsets+'/offsets_{}_data.p'.format(which)
    if which == 'left':
        R = wrist_map_c2l
    else:
        R = wrist_map_c2r
    num_offsets = 0

    while True:
        # Just to give an overall view before grabbing needle. OK to only see the left image.
        arm.open_gripper(50)
        print("\nNow starting new offset with {} offsets so far...".format(num_offsets))
        name = "Left image w/{} offsets so far. MOVE DVRK, press any key".format(num_offsets)
        U.call_wait_key( cv2.imshow(name, d.left['raw']) )
        arm.close_gripper()
        pos,rot = U.pos_rot_arm(arm, nparrays=True)
        pos[2] += 0.010
        U.move(arm, pos, rot)

        # The user now clicks stuff. LEFT CAMERA, then RIGHT CAMERA.
        click_stuff('left')
        click_stuff('right')

        # Compute data to store. We stored left tip, left grip, right tip, right grip.
        pos,rot = U.pos_rot_arm(arm, nparrays=True)
        assert len(CENTERS) % 4 == 0, "Error, len(CENTERS): {}".format(len(CENTERS))
        camera_tip  = U.camera_pixels_to_camera_coords(CENTERS[-4], CENTERS[-2], nparrays=True)
        camera_grip = U.camera_pixels_to_camera_coords(CENTERS[-3], CENTERS[-1], nparrays=True)

        # Map stuff to stuff. The `h` in `xyz_h` refers to homogeneous coordinates.
        ct_h = np.concatenate( (camera_tip,np.ones(1)) )
        cg_h = np.concatenate( (camera_grip,np.ones(1)) )
        base_t = R.dot(ct_h)
        base_g = R.dot(cg_h)
        camera_dist = np.linalg.norm( camera_tip-camera_grip )
        base_dist   = np.linalg.norm( base_t-base_g )

        # Bells and whistles. Note 
        base = 'base_'+which+'_'
        info = {}
        info['pos_dvrk'] = pos
        info['rot_dvrk'] = rot
        info['camera_tip'] = camera_tip
        info['camera_grip'] = camera_grip
        info[base+'tip'] = base_t
        info[base+'grip'] = base_g
        info['camera_dist'] = camera_dist 
        info[base+'dist'] = base_dist

        num_offsets += 1
        U.store_pickle(fname=pname, info=info, mode='a')
        num_before_this = U.get_len_of_pickle(pname)

        print("Computed and saved {} offset vectors in this session".format(num_offsets))
        print("We have {} items total (including prior sessions)".format(num_before_this))
        print("  pos, rot = {}, {}".format(pos, rot))
        print("  for tip, CENTER coords (left,right): {}, {}".format(CENTERS[-4], CENTERS[-2]))
        print("  for grip, CENTER coords (left,right): {}, {}".format(CENTERS[-3], CENTERS[-1]))
        print("  camera_tip:  {}".format(camera_tip))
        print("  camera_grip: {}".format(camera_grip))
        print("  base_{}_tip:  {}".format(which, base_t))
        print("  base_{}_grip: {}".format(which, base_g))
        print("  camera_dist: {}".format(camera_dist))
        print("  base_dist:   {}".format(base_dist))


def evaluate(arm, d, args, wrist_map_c2l, wrist_map_c2r, wrist_map_l2r):
    """ Stage 2. 
    
    Testing stage. Move the dVRK end-effector to various locations and once
    again explicitly record needle tips. This time we evaluate the accuracy.
    This stage is similar to the second one except we don't compute offset
    vectors.  I will need to cheat and explicitly compute my `\phi`s so that we
    know which offset vectors to use.
    """
    assert os.path.isfile(path_offsets)
    with open(path_offsets, 'r') as ff:
        offsets = pickle.load(f)


if __name__ == "__main__":
    pp = argparse.ArgumentParser()
    pp.add_argument('--arm', type=int, default=1, help='Must be 1 or 2.')
    pp.add_argument('--stage', type=int, help='Must be 1 or 2.')
    pp.add_argument('--path_calib', type=str, default='calibration/')
    pp.add_argument('--path_offsets', type=str, default='scripts/offsets')
    args = pp.parse_args()
    print("Did you remember to remove the old offsets file if we're ignoring that data?")

    # Bells and whistles.
    arm1, arm2, d = U.init()
    time.sleep(1)
    wrist_map_c2l = U.load_pickle_to_list(args.path_calib+'wrist_map_c2l.p', squeeze=True)
    wrist_map_c2r = U.load_pickle_to_list(args.path_calib+'wrist_map_c2r.p', squeeze=True)
    wrist_map_l2r = U.load_pickle_to_list(args.path_calib+'wrist_map_l2r.p', squeeze=True)
    if args.arm == 1:
        arm = arm1
    elif args.arm == 2:
        arm = arm2
    else:
        raise ValueError()
    debug_images(d, args)

    if args.stage == 1:
        offsets(arm, d, args, wrist_map_c2l, wrist_map_c2r, wrist_map_l2r)
    elif args.stage == 2:
        evaluate(arm, d, args, wrist_map_c2l, wrist_map_c2r, wrist_map_l2r)
    else:
        raise ValueError("illegal args.stage = {}".format(args.stage))
