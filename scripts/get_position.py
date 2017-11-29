"""
Try and see good ranges of yaw/pitch/roll for some target needle position.

- Put the needle on the surface, get the gripper to it (but keep it open).
- Grip the needle (in a "reasonable" way, e.g. from the ground), this is needed
  since different yaw/pitch/rolls could easily result in the same needle shape.
- Get a piece of paper, put it under the gripper
- Now try and find yaw, pitch, and roll values for the gripper, which will align
  the needle with the shape in that piece of paper.
- Repeat to get representative ranges for yaw, pitch, and roll values.

Stay positive!!  Observations:

- We need SNAP on one particular side of the needle. If we view the needle as an
  umbrella, then SNAP goes on the outside.
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

# Possible/ideal starts:
starts = [
    ([0.04671, 0.02062, -0.13247], [94.25816, -3.75545, 152.21161]),
]

# Possible/ideal targets:
targets = [
    ([0.05087, 0.03142, -0.12227], [137.12579, 65.46415, 143.77501]),
    ([0.04964, 0.03091, -0.12575], [116.96745, 65.58404, 147.46808]),
]


def test(arm):
    """ Use to see if we can get it to move it to roughly the right target. """
    spos, srot = starts[0]
    tpos, trot = targets[1]

    U.move(arm, spos, srot)
    print("moved: {}".format(U.pos_rot_arm(arm, nparrays=True)))
    time.sleep(2)

    spos[2] += 0.02
    U.move(arm, spos, srot)
    print("moved: {}".format(U.pos_rot_arm(arm, nparrays=True)))
    time.sleep(2)

    U.move(arm, tpos, trot)
    print("moved: {}".format(U.pos_rot_arm(arm, nparrays=True)))
    time.sleep(2)

    trot[1] = 80
    U.move(arm, tpos, trot)
    print("moved: {}".format(U.pos_rot_arm(arm, nparrays=True)))


def explore(arm):
    """ Use this to extract positions after we move the dvrk. """
    pos, rot = U.pos_rot_arm(arm, nparrays=True)
    print("pos, rot: {}, {}".format(pos, rot))
    pos[2] += 0.02
    U.move(arm, pos, rot)
    print("pos, rot: {}, {}".format(pos, rot))


if __name__ == "__main__":
    arm1, arm2, d = U.init()
    arm1.close_gripper()

    #explore(arm1)
    #test(arm1)

    print("start {}".format(U.pos_rot_arm(arm1, nparrays=True)))

    # this will fail unless it's starting near that position
    pos, rot = [0.039, 0.045, -0.074], [45.7, 64.7, 62.2]
    U.move(arm1, pos, rot)
    print(U.pos_rot_arm(arm1, nparrays=True))
    time.sleep(2)

    #rot = [79.14, 45.98, 57.77] # fails a lot
    rot = [60., 50., 60.]
    U.move(arm1, pos, rot)
    print(U.pos_rot_arm(arm1, nparrays=True))
