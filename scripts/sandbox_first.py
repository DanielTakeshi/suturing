"""
My first open loop for suturing. Just assume fixed, pre-determined robot
coordinates and try to get SOMETHING working.
"""
from autolab.data_collector import DataCollector
from dvrk.robot import *
import cv2
import numpy as np
import os
import pickle
import sys
import tfx
import time
import utils as U
np.set_printoptions(suppress=True)

# Change these ...
#TESTING = False
TESTING = True


if __name__ == "__main__":
    arm1, arm2, d = U.init()
    arm1.close_gripper()
    fname = 'tmp/tmp_pos.p'

    if not TESTING:
        # Have the human press key to continue, escape to terminate.
        key = -1
        positions = []
        stored = 0

        while key not in U.ESC_KEYS:
            key = U.call_wait_key(
                    cv2.imshow("ESC to exit, stored {}.".format(stored), d.left_image), 
                    exit=False
            )
            pos,rot = U.get_pos_rot_from_arm(arm1)
            positions.append((pos,rot))
            print("Appended: {}".format((pos,rot)))
            stored += 1

        if stored > 0:
            U.store_pickle(fname, positions)

    else:
        positions = U.load_pickle_to_list(fname) 
        print("loaded {} positions".format(len(positions)))
        arm1.close_gripper()

        for i,(pos,rot) in enumerate(positions):
            U.move(arm1, pos, rot)
            arm1.close_gripper()
            print("({}) Moved to {},{}".format(i,pos,rot))
