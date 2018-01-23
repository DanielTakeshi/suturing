"""
Follow my notes and get an analytic model of the needle. What this means in our
context is that I'm trying to control the robot so that the needle tip ends up
in the correct location.  This should be a self-contained script purely for
testing the analytic model, and not to be used in practice. 

Call this code in stages, so `python analytic_model.py --stage X` where:

    - X == 1: do a one-plane calibration, similar to what I did in the ICRA 2018
      paper. We need calibration to get correct offset vectors since we
      determine them w.r.t. the camera frames. I don't *think* we need to vary
      the height, I think just doing it on one height is enough, because with
      other heights, the tool frame is going to move along with it.

      The major concern I have is if this will work with a fixed dVRK but moving
      the yaw pitch and rolls, since the three angles don't seem to be quite
      standard. But that's what the testing stage should reveal (the X==3 case).

      WAIT ... I was calibrating with a fixed rotation. Ahh ... will that change
      things?

    - X == 2: before doing this, manipulate the dVRK with SNAP so that the
      needle is at the maximum value of $\phi$, or when the gripper is gripping
      it at the furthest point from the needle. We also need to get the needle
      on a flat (foam) surface to make the task easier. 
      
      This stage iteratively stops to let the user manipulate the needle forward
      ever so slightly. At each 'nudge' by the user, the code stops and outputs
      images. We click on the location of the needle tip by dragging a box
      around it. Then our calibration converts that to a position w.r.t. the
      BASE frame. But note that we can simply query our robot position, also
      w.r.t.  the base frame. This gives us offset vectors.

    - X == 3: testing stage. Now we move the dVRK end-effector to various
      configurations and once again explicitly record needle tips. This time we
      evaluate the accuracy. This stage is similar to the second one except we
      don't compute offset vectors. 
      
      I will need to cheat and explicitly compute my `\phi`s so that we know
      which offset vectors to use.

(c) 2018 by Daniel Seita
"""
from autolab.data_collector import DataCollector
from dvrk import robot
import argparse, cv2, os, pickle, sys, tfx, time
import numpy as np
import utils as U
np.set_printoptions(suppress=True, precision=5)


def calibrate(arm1, arm2, d):
    """ Stage 1. """


def offsets(arm1, arm2, d, path_calib):
    """ Stage 2. """
    assert os.path.isfile(path_calib)
    with open(path_calib, 'r') as ff:
        calib = pickle.load(f)
    num_offsets = 0

    while True:
        key = U.call_wait_key( cv2.imshow(d['left']) )
        print("currently computed {} offset vectors so far".format(num_offsets))
        num_offsets += 1


def evaluate(arm1, arm2, d, path_offsets):
    """ Stage 3. """
    assert os.path.isfile(path_offsets)
    with open(path_offsets, 'r') as ff:
        offsets = pickle.load(f)


if __name__ == "__main__":
    pp = argparse.ArgumentParser()
    pp.add_argument('--stage', type=int, help='Must be 0, 1 or 2.')
    pp.add_argument('--path_calib', type=str, default='files/analytic_calib.p')
    pp.add_argument('--path_offsets', type=str, default='files/analytic_offsets.p')
    args = pp.parse_args()

    arm1, arm2, d = U.init()
    arm1.close_gripper()

    if args.stage == 1:
        calibrate(arm1, arm2, d)
    elif args.stage == 2:
        offsets(arm1, arm2, d, args.path_calib)
    elif args.stage == 3:
        evaluate(arm1, arm2, d, args.path_offsets)
    else:
        raise ValueError("illegal args.stage = {}".format(args.stage))
