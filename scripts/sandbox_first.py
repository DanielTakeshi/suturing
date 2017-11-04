"""
My first open loop for suturing. Just assume fixed, pre-determined robot
coordinates and try to get SOMETHING working.

Specifically, what this does is pause the code and show an image of the
workspace. The human should move the robot arm to the correct pose (directly
touching it). Then press any key other than ESC and this will record the
position. Keep doing this until as many positions are desired, and then press
ESC. The positions are stored in a pickle file, so that if we run it later with
the `--test` argument set to 1, we'll move the robot arm to those positions.

The pickle files are stored in `tmp/`. If you want to save a pickle file, then
you can manually move it and then load it in via a parser argument.

Observations from my runs:

    - Probably easiest for the dvrk to grip the dvrk at an angle. Otherwise it
      loses grip easily.
    - Going to be very difficult with the suturing phanton we have. When the
      needle pierces the phantom, the dVRK's wrist seems to lose control, the
      wrist moves but the end-effectors don't (like how a human would).
    - Let's get a fixed position and just get it to follow a nice circular path.
      Use DART maybe.

Usage example:

    python scripts/sandbox_first.py --test 0 
    python scripts/sandbox_first.py --test 1 --test_method 1
    python scripts/sandbox_first.py --test 1 --test_method 0 --fname tmp/tmp_pos_revised.p

First have humans provide the demonstrations. Then roll it out but the human
corrects for it (run this several times), Then roll it out entirely open loop.
The `fname` is needed for the second argument to use the revised positions.
"""

from autolab.data_collector import DataCollector
from dvrk.robot import *
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


def test(arm1, fname, test_method):
    """ Roll out the open loop policy. No human intervention. 
    
    test_method
    -----------
        0: roll entirely open-loop, no human intervention.
        1: roll each time step, then human provides correction (saves new file).
    """
    positions = U.load_pickle_to_list(fname) 
    print("loaded {} positions from {}".format(len(positions), fname))
    if test_method == 0:
        print("We're going to run entirely open-loop.")
    elif test_method == 1:
        print("NOTE! You'll have to provide corrections after each movement.")
        revised_positions = []
    arm1.close_gripper()

    for i,(pos,rot) in enumerate(positions):
        U.move(arm1, pos, rot)
        arm1.close_gripper()
        real_pos,real_rot = U.get_pos_rot_from_arm(arm1, nparrays=True)
        print("\n({}) Target position: {},{}".format(i,pos,rot))
        print("    Actual position: {},{}".format(real_pos,real_rot))

        if test_method == 1:
            string = "Now correct the position as needed, then press any key,"+ \
                    " other than ESC (which will terminate the entire program)."
            U.call_wait_key(cv2.imshow(string, d.left_image), exit=True)
            revised_pos,revised_rot = U.get_pos_rot_from_arm(arm1, nparrays=True)
            revised_positions.append( (revised_pos,revised_rot) )
            print("    Revised position: {},{}".format(revised_pos,revised_rot))
        else:
            time.sleep(2)

    if test_method == 1:
        new_fname = fname[:-2] + '_revised.p'
        print("Storing {} positions in file {}".format(len(revised_positions), new_fname))
        U.store_pickle(new_fname, revised_positions)


def get_data(arm1, fname):
    """ Have the human press key to continue, escape to terminate. """
    key = -1
    positions = []
    stored = 0

    while True:
        key = U.call_wait_key(
                cv2.imshow("ESC to exit and skip this; stored {} so far.".format(
                        stored), d.left_image), 
                exit=False
        )
        if key in U.ESC_KEYS:
            break
        pos,rot = U.get_pos_rot_from_arm(arm1, nparrays=True)
        positions.append((pos,rot))
        print("Appended: {}".format((pos,rot)))
        stored += 1

    if stored > 0:
        print("Storing {} positions in file {}".format(stored, fname))
        U.store_pickle(fname, positions)


if __name__ == "__main__":
    pp = argparse.ArgumentParser()
    pp.add_argument('--test', type=int)
    pp.add_argument('--test_method', type=int, default=0) # See later
    pp.add_argument('--fname', type=str, default=None)
    args = pp.parse_args()
    assert args.test in [0,1], "Did you forget to set args.test?"
    assert args.test_method in [0,1]

    fname = 'tmp/tmp_pos.p' if args.fname is None else args.fname
    arm1, arm2, d = U.init()
    arm1.close_gripper()

    if args.test == 1:
        test(arm1, fname, args.test_method)
    else:
        get_data(arm1, fname)
