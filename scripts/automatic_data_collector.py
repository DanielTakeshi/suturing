"""
Run the robot with the needles gripped and automatically collect data.  See the
main method for the hyperparameters.

Usage: run in a two-stage process. Run this the first time for some manual
intervention to determine the boundaries of the workspace and the height bounds.
The second time is when the real action begins and the trajectories can be run.
The third run is for data cleaning.

    STAGE 1: Make sure the four corners are in areas near the center of the
    camera. We don't want to emphasize the edges, yet.

    STAGE 2: to vary the gripper orientation, start the second stage by having
    the gripper grip the needle on a flat surface. Then it moves and starts off.
    This must be done after each set, which may consist of several trajectories.

        ALSO, to make sure that the dVRK doesn't crash, we should limit the
        magnitude of change.

        TODO: add more constraints, so that we're in the more likely areas where
        the dVRK will want to "enter" the gripper? Some of the default angle
        configurations won't have this correct setup.

    STAGE 3: in progress ...

Note I: the code saves at the end of each trajectory, so it's robust to cases
when the dVRK might fail (e.g., that MTM reading error).

NOTE II: when running this for real, be sure to feed (i.e. `tee`) the output
into a file for my own future reference, or have this code write to a file.

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


class AutoDataCollector:

    def __init__(self, args):
        self.d = args['d']
        self.arm = args['arm']
        self.num_trajs = args['num_trajs']
        self.rots_per_stoppage = args['rots_per_stoppage']
        self.interpolation_interval = args['interpolation_interval']
        self.require_positive_roll = args['require_positive_roll']
        self.require_negative_roll = args['require_negative_roll']

        self.info = pickle.load( open(args['guidelines_dir'], 'r') )
        self.z_alpha = self.info['z_alpha']
        self.z_beta  = self.info['z_beta']
        self.z_gamma = self.info['z_gamma']

        self.z_offset = args['z_offset']
        self.orien = ['lr','ll','ul','ur']

        print("\nHere's what we have in our `self.info`:")
        keys = sorted(self.info.keys())
        for key in keys:
            print("{:20} ==> {}".format(key, self.info[key]))
        print("")
 

    def _get_z_from_xy_values(self, x, y):
        """ We fit a plane. """
        return (self.z_alpha * x) + (self.z_beta * y) + self.z_gamma

    
    def _get_thresholded_image(self, image, left):
        """ Use this to get thresholded images from RGB (not BGR) `image`. 
       
        This should produce images that are easy enough for contour detection code 
        later, which should detect the center of the largest contour and deduce that
        as the approximate location of the end-effector.
    
        We assume the color target is red, FYI, and that it's RGB->HSV, not BGR->HSV.
        """
        # Requires some tuning and inspecting stuff from the two cameras.
        if left:
            lower = np.array([110, 90, 90])
            upper = np.array([180, 255, 255])
        else:
            lower = np.array([110, 70, 70])
            upper = np.array([180, 255, 255])
    
        #image = cv2.medianBlur(image, 9)
        image = cv2.bilateralFilter(image, 7, 13, 13)
    
        hsv   = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask  = cv2.inRange(hsv, lower, upper)
        res   = cv2.bitwise_and(image, image, mask=mask)
        res   = cv2.medianBlur(res, 9)
        return res


    def _sample_yaw(self, old_yaw=None):
        """ Samples a valid yaw, with mechanism to prevent large changes. """
        yaw = np.random.uniform(low=self.info['min_yaw'], high=self.info['max_yaw'])
        if old_yaw is not None:
            while (np.abs(yaw-old_yaw) > 150): # Prevents large extremes
                yaw = np.random.uniform(low=self.info['min_yaw'], high=self.info['max_yaw'])
        return yaw


    def _sample_pitch(self, old_pitch=None):
        """ Samples a valid pitch. """
        return np.random.uniform(low=self.info['min_pitch'], high=self.info['max_pitch'])


    def _sample_roll(self, old_roll=None):
        """ Samples a valid roll, & handle roll being negative. """
        if self.require_positive_roll:
            roll = np.random.uniform(low=self.info['min_pos_roll'], high=self.info['max_pos_roll']) 
        elif self.require_negative_roll:
            roll = np.random.uniform(low=self.info['min_neg_roll'], high=self.info['max_neg_roll'])
        else:
            roll = np.random.uniform(low=-180, high=180) 
            while (self.info['roll_neg_ubound'] < roll < self.info['roll_pos_lbound']):
                roll = np.random.uniform(low=-180, high=180) 
        return roll

    
    def _sample_safe_rotation(self, prev_rot):
        """ 
        Need to tune these limit ranges!! Sample rotation repeatedly until all
        the safety checks pass.
        """
        ylim = 40
        plim = 20
        rlim = 30
        all_ratio = 0.66

        prev_roll = prev_rot[2]
        if (not self.require_positive_roll) and (not self.require_negative_roll):
            if prev_roll < 0:
                prev_roll += 360.
        total_rejected = 0

        while True:
            possible_rot = [self._sample_yaw(), self._sample_pitch(), self._sample_roll()]
            # ------------------------------------------------------------------
            # Ah, but roll could be from (-180,-K) u (K,180). Thus, we add 360
            # to the negative values to make it (K,180) to (180,360-K). If the
            # previous roll was 170 deg., and current one is -180, then it's
            # really close by 10 degrees. If current is -140, should be farther
            # ... which it is as 360-140=220, so the gap is 50 degrees.
            # ------------------------------------------------------------------
            curr_roll = possible_rot[2]
            if (not self.require_positive_roll) and (not self.require_negative_roll):
                if curr_roll < 0:
                    curr_roll += 360.

            y_diff = abs(possible_rot[0] - prev_rot[0])
            p_diff = abs(possible_rot[1] - prev_rot[1])
            r_diff = abs(prev_roll - curr_roll)
            y_cond = y_diff < ylim
            p_cond = p_diff < plim
            r_cond = r_diff < rlim
            all_cond = (y_diff + p_diff + r_diff) < (ylim + plim + rlim) * all_ratio

            if y_cond and p_cond and r_cond:
                break
            else:
                random_rotation_str = ["%.2f" % v for v in possible_rot]
                #print("    prev, curr rolls: {} and {}".format(prev_roll, curr_roll))
                #print("    _not_ accepting sampled rot:  {} due to y,p,r={},{},{}".format(
                #    random_rotation_str, y_cond, p_cond, r_cond))
                total_rejected += 1
 
        print("    total rejected sampled rot: {}".format(total_rejected))
        return possible_rot


    def _save_images(self, this_dir, num, rr):
        """ Saves me some typing. """
        left_th  = self._get_thresholded_image(self.d.left['raw'].copy(),  left=True)
        right_th = self._get_thresholded_image(self.d.right['raw'].copy(), left=False)

        # TODO this is where I need to fix the code so that I can get processed
        # images for needles, or else try this again.
        # left_th, l_center  = self._get_contour(left_th,  left=True)
        # right_th, r_center = self._get_contour(right_th, left=False)

        cv2.imwrite(this_dir+'left/'+num+'_rot'+rr+'_left.jpg',         self.d.left['raw'])
        cv2.imwrite(this_dir+'right/'+num+'_rot'+rr+'_right.jpg',       self.d.right['raw'])
        cv2.imwrite(this_dir+'left_th/'+num+'_rot'+rr+'_left_th.jpg',   left_th)
        cv2.imwrite(this_dir+'right_th/'+num+'_rot'+rr+'_right_th.jpg', right_th)
        # return l_center, r_center
        return (0,0), (0,0)


    def _move_to_random_home_position(self):
        """ Moves to one of the home positions we have set up. 
        
        One change: I adjust the angles to be in line with what we have. The
        reason is that we'll keep referring back to this angle when we move, and
        I want it to be inside the actual ranges of yaw, pitch, and roll we're
        considering. Hence, the `better_rotation` that's here. So, we move to
        the random home position, and just fix up the rotation.
        """
        index = np.random.randint(len(self.orien))
        home_pos = self.info['pos_'+self.orien[index]]
        home_rot = self.info['rot_'+self.orien[index]]
        U.move(self.arm, home_pos, home_rot)
        #better_rotation = [self._sample_yaw(), self._sample_pitch(), self._sample_roll()]
        better_rotation = self._sample_safe_rotation(home_rot)
        U.move(self.arm, home_pos, better_rotation)


    def collect_trajectories(self):
        """ Runs the robot and collects `self.num_trajs` trajectories. 
        
        0. Let the gripper grip the needle; there is natural variability which is OK.
        1. Move to one of the pre-selected home positions (from stage 1) at random.
        2. Choose a random location of (x,y,z) that is within legal and safe range.
        3. Make the arm to move there, but stop it periodically along its trajectory.
        4. For each point in its trajectory, randomly make it rotate to certain spots.
        5. Keep collecting and saving left AND right camera images of this process.
        6. Each trajectory is saved in its own sub-directory of images.

        The trajectories are saved according to index, so we can (and should)
        run this many times and we won't be destroying old data.

        AT minimum, save the NORMAL/RAW camera views, because we can always do
        our own image preprocessing afterwards.

        There are `time.sleep(...)` commands since there's a delay when camera
        images are updated. TODO check if this affects anything...
        """
        traj_dirs = [x for x in os.listdir('traj_collector') if 'traj' in x]
        traj_index = len(traj_dirs)
        print("\nCollecting trajectories. Starting index: {}\n".format(traj_index))
        
        for tt in range(self.num_trajs):
            # Get directories/stats set up; move to a random home position.
            this_dir = 'traj_collector/traj_'+str(traj_index).zfill(4)+'/'
            os.makedirs(this_dir)
            os.makedirs(this_dir+'left/')
            os.makedirs(this_dir+'right/')
            intervals_in_traj = 0
            traj_poses = []
            self._move_to_random_home_position()
            time.sleep(5)

            # Pick a safe target position. 
            xx = np.random.uniform(low=self.info['min_x'], high=self.info['max_x'])
            yy = np.random.uniform(low=self.info['min_y'], high=self.info['max_y'])
            zz = self._get_z_from_xy_values(xx, yy)
            target_position = [xx, yy, zz + self.z_offset] 
            print("\n\nTrajectory {}, target position: {}".format(traj_index, target_position))

            # ------------------------------------------------------------------
            # Follows the `linear_interpolation` movement code to take incremental sets.
            interval   = 0.0001
            start_vect = self.arm.get_current_cartesian_position()

            # Get a list with the yaw, pitch, and roll from the _starting_ position.
            _, current_rot = U.pos_rot_cpos(start_vect)

            # If calling movement code, `end_vect` would be the input "tfx frame."
            end_vect     = tfx.pose(target_position, tfx.tb_angles(*current_rot))
            displacement = np.array(end_vect.position - start_vect.position)    

            # Total interval present between start and end poses
            tinterval = max(int(np.linalg.norm(displacement)/ interval), 50)    
            print("Number of intervals: {}".format(tinterval))

            for ii in range(0, tinterval, self.interpolation_interval):
                # SLERP interpolation from tfx function (from `dvrk/robot.py`).
                mid_pose = start_vect.interpolate(end_vect, (ii+1.0)/tinterval)   
                self.arm.move_cartesian_frame(mid_pose, interpolate=True)

                # --------------------------------------------------------------
                # Back to my stuff. Note that `frame` = `mid_pose` (in theory, at least).
                time.sleep(4)
                print("\ninterval {} of {}, mid_pose: {}".format(ii+1, tinterval, mid_pose))
                frame = self.arm.get_current_cartesian_position() # The _actual_ pose.
                this_pos, this_rot = U.pos_rot_cpos(frame)
                previous_rotation = this_rot

                # After moving there (keeping rotation fixed) we record information.
                num = str(intervals_in_traj).zfill(3) # Increments by 1 each non-rotation movement.
                l_center, r_center = self._save_images(this_dir, num, '0')
                traj_poses.append( (frame,l_center,r_center) )
           
                for rr in range(1, self.rots_per_stoppage+1):
                    # Pick a random but SAFE rotation, and move there.
                    random_rotation = self._sample_safe_rotation(previous_rotation)
                    random_rotation_str = ["%.2f" % v for v in random_rotation]
                    print("    rot {}, _final_ target rot:  {}".format(rr, random_rotation_str))
                    
                    # Now actually MOVE to this rotation, w/"same" position.
                    U.move(self.arm, this_pos, random_rotation)
                    time.sleep(4)
                    frame = self.arm.get_current_cartesian_position()
                    print("    rot {}, actual pose: {}".format(rr, frame))

                    # Set our previous rotation to be the ACTUAL rotation.
                    _, this_rot_1 = U.pos_rot_cpos(frame)
                    previous_rotation = this_rot_1

                    # Record information.
                    l_center, r_center = self._save_images(this_dir, num, str(rr))
                    traj_poses.append( (frame,l_center,r_center) )

                # Back to the original rotation; I think this will make it work better.
                U.move(self.arm, this_pos, this_rot)
                intervals_in_traj += 1

            # Finished with this trajectory! Save here and go to next directory.
            print("Finished with trajectory. len(traj_poses): {}".format(len(traj_poses)))
            pickle.dump(traj_poses, open(this_dir+'traj_poses_list.p', 'w'))
            traj_index += 1


def collect_guidelines(args, arm, d):
    """ Gather statistics about the workspace on how safe we can set things.
    Save things in a pickle file specified by the `directory` parameter. Click
    the ESC key to exit the program and restart if I've made an error. BTW, the
    four poses we collect will be the four "home" positions that I use later,
    though with more z-coordinate offset.

    Some information:

        `yaw` must be limited in [-180,180]  # But actually, [-90,90] is OK.
        `pitch` must be limited in [-50,50]  # I _think_ ...
        `roll` must be limited in [-180,180] # I think ...

    Remember, if I change the numbers, it doesn't impact the code until
    re-building `guidelines.p`!!
    """
    # Add stuff we should already know, particularly the rotation ranges.
    info = {}
    info['min_yaw']   = 30
    info['max_yaw']   = 150
    info['min_pitch'] = 40
    info['max_pitch'] = 80

    # Roll is annoying because of the values the dVRK provides.
    info['roll_neg_ubound'] = -140 # (-180, roll_neg_ubound)
    info['roll_pos_lbound'] =  140 # (roll_pos_lbound, 180)
    info['min_pos_roll'] =  40
    info['max_pos_roll'] = 180
    info['min_neg_roll'] = -180
    info['max_neg_roll'] = -150

    # Move arm to positions to determine approximately safe ranges for x,y,z
    # values. All the `pos_{lr,ll,ul,ur}` are in robot coordinates.
    U.call_wait_key(cv2.imshow("Left camera (move to lower right corner now!)", d.left['raw']))
    info['pos_lr'], info['rot_lr'] = U.pos_rot_arm(arm)
    U.call_wait_key(cv2.imshow("Left camera (move to lower left corner now!)", d.left['raw']))
    info['pos_ll'], info['rot_ll'] = U.pos_rot_arm(arm)
    U.call_wait_key(cv2.imshow("Left camera (move to upper left corner now!)", d.left['raw']))
    info['pos_ul'], info['rot_ul'] = U.pos_rot_arm(arm)
    U.call_wait_key(cv2.imshow("Left camera (move to upper right corner now!)", d.left['raw']))
    info['pos_ur'], info['rot_ur'] = U.pos_rot_arm(arm)

    # So P[:,0] is a vector of the x's, P[:,1] vector of y's, P[:,2] vector of z's.
    p_lr = np.squeeze(np.array( info['pos_lr'] ))
    p_ll = np.squeeze(np.array( info['pos_ll'] ))
    p_ul = np.squeeze(np.array( info['pos_ul'] ))
    p_ur = np.squeeze(np.array( info['pos_ur'] ))
    P = np.vstack((p_lr, p_ll, p_ul, p_ur))

    # Get ranges. This is a bit of a heuristic but generally good I think.
    info['min_x'] = np.min( [p_lr[0], p_ll[0], p_ul[0], p_ur[0]] )
    info['max_x'] = np.max( [p_lr[0], p_ll[0], p_ul[0], p_ur[0]] )
    info['min_y'] = np.min( [p_lr[1], p_ll[1], p_ul[1], p_ur[1]] )
    info['max_y'] = np.max( [p_lr[1], p_ll[1], p_ul[1], p_ur[1]] )

    # For z, we fit a plane. See https://stackoverflow.com/a/1400338/3287820
    # Find (alpha, beta, gamma) s.t. f(x,y) = alpha*x + beta*y + gamma = z.
    A = np.zeros((3,3)) # Must be symmetric!
    A[0,0] = np.sum(P[:,0] * P[:,0])
    A[0,1] = np.sum(P[:,0] * P[:,1])
    A[0,2] = np.sum(P[:,0])
    A[1,0] = np.sum(P[:,0] * P[:,1])
    A[1,1] = np.sum(P[:,1] * P[:,1])
    A[1,2] = np.sum(P[:,1])
    A[2,0] = np.sum(P[:,0])
    A[2,1] = np.sum(P[:,1])
    A[2,2] = P.shape[0]

    b = np.array(
            [np.sum(P[:,0] * P[:,2]), 
             np.sum(P[:,1] * P[:,2]), 
             np.sum(P[:,2])]
    )

    x = np.linalg.inv(A).dot(b)
    info['z_alpha'] = x[0]
    info['z_beta']  = x[1]
    info['z_gamma'] = x[2]

    # Sanity checks before saving stuff.
    assert info['min_x'] < info['max_x']
    assert info['min_y'] < info['max_y']
    assert P.shape == (4,3)

    print("\nThe key/val pairings for {}.".format(args.directory))
    keys = sorted(info.keys())
    for key in keys:
        print("{:20} ==> {}".format(key, info[key]))
    print("")
    print("P:\n{}".format(P))
    print("A:\n{}".format(A))
    print("x:\n{}".format(x))
    print("b:\n{}".format(b))
    U.store_pickle(fname=args.directory, info=info, mode='w') 


if __name__ == "__main__": 
    pp = argparse.ArgumentParser()
    pp.add_argument('--stage', type=int, help='Must be 0, 1 or 2.')
    pp.add_argument('--directory', type=str, default='traj_collector/guidelines.p')
    pp.add_argument('--num_trajs', type=int, default=1) # TODO: think, depends on needle grip
    pp.add_argument('--rots_per_stoppage', type=int, default=3)
    pp.add_argument('--interpolation_interval', type=int, default=20)
    args = pp.parse_args()
    assert args.stage in [0,1,2]

    arm1, arm2, d = U.init(sleep_time=2)
    arm1.close_gripper()

    if args.stage == 0:
        print("Stage 0, manual collection of safe ranges for auto collection.")
        collect_guidelines(args, arm1, d)

    elif args.stage == 1:
        print("Stage 1, auto data collection. DID THE GRIPPER GRIP THE NEEDLE?")
        adc_args = {}
        adc_args['d'] = d
        adc_args['arm'] = arm1
        adc_args['z_offset'] = 0.010 # TODO is this needed?
        adc_args['require_positive_roll'] = True  # TODO double check
        adc_args['require_negative_roll'] = False # TODO double check
        adc_args['num_trajs'] = args.num_trajs
        adc_args['rots_per_stoppage'] = args.rots_per_stoppage
        adc_args['guidelines_dir'] = args.directory
        adc_args['interpolation_interval'] = args.interpolation_interval
        ADC = AutoDataCollector(adc_args)
        ADC.collect_trajectories()

    elif args.stage == 2:
        # Perform data cleaning
        print("Stage 2, data cleaning on the data.")
        print("TODO: not implemented yet.")
