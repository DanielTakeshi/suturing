"""
Used only to test moving to certain locations to ensure that coordinates are OK.
"""
from autolab.data_collector import DataCollector
from dvrk.robot import *
import argparse, cv2, os, pickle, sys, tfx, time
import numpy as np
import utils as U
np.set_printoptions(suppress=True, precision=5)


x7 = [(0.005, 0.028, -0.103), (32.1, 73.7, -56.8)]
x11 = [(0.005, 0.028, -0.112), (32.1, 73.7, -56.8)]
x9 = [(0.027, 0.033, -0.123), (-11.8, 30.8, -103.0)]
x12 = [(0.032, 0.037, -0.125), (-0.5, 8.0, -86.7)]

y0 = [(-0.140, 0.077, -0.076), (-166.4, 27.1, 134.2)]
y1 = [(-0.139, 0.081, -0.078), (178.7, 28.8, 132.0)]
y1 = [(-0.138, 0.089, -0.080), (178.4, 29.1, 126.5)]
y1 = [(-0.136, 0.081, -0.075), (-179.5, 27.8, 127.8)]
y1 = [(-0.137, 0.082, -0.077), (-179.6, 28.3, 127.3)]
y0 = [(-0.140, 0.080, -0.077), (-178.5, 28.1, 127.6)]
y1 = [(-0.140, 0.082, -0.079), (-179.1, 28.2, 128.0)]

z0 = [(0.032, 0.030, -0.126), (-1.1, 11.4, -89.4)]
z1 = [(-0.137, 0.086, -0.076), (0.8, 8.2, -83.8)]
z1 = [(0.041, 0.028, -0.092), (-3.3, 19.4, -74.8)]

w0 = [(-0.122, 0.068, -0.060), (-179.8, 25.7, 130.4)]
w1 = [(-0.121, 0.062, -0.050), (-178.0, 23.6, 131.2)]

w0 = [(-0.125, 0.072, -0.066), (179.3, 21.7, 143.1)]
w1 = [(-0.125, 0.068, -0.065), (-178.8, 20.9, 143.9)]
w1 = [(-0.131, 0.069, -0.063), (-178.4, 19.0, 142.8)]
w1 = [(-0.130, 0.067, -0.063), (-178.2, 19.3, 138.2)]
w1 = [(-0.132, 0.068, -0.063), (-179.0, 18.9, 137.7)]

v0 = [(-0.103, 0.039, -0.042), (-170.9, 21.1, 130.5)]
u0 = [( 0.038, 0.032, -0.114), (-1.8, 16.2, -79.8)]



if __name__ == "__main__":
    arm1, arm2, d = U.init()
    arm1.close_gripper()
    arm2.close_gripper()

    print(U.pos_rot_arm(arm1, nparrays=True))
    print(U.pos_rot_arm(arm2, nparrays=True))

    U.move(arm1, x7[0],  x7[1])
    U.move(arm1, x11[0], x11[1])
    U.move(arm1, x9[0],  x9[1])
    U.move(arm1, x12[0], x12[1])

    U.move(arm2, y0[0], y0[1])
    arm2.open_gripper(70)
    time.sleep(2)
    U.move(arm2, y1[0], y1[1])
    arm2.open_gripper(-10)
    time.sleep(2)

    arm1.open_gripper(70)
    time.sleep(2)
    U.move(arm1, z0[0], z0[1])
    U.move(arm1, z1[0], z1[1])

    U.move(arm2, w0[0], w0[1])
    time.sleep(1)
    U.move(arm2, w1[0], w1[1])

    arm1.open_gripper(-10)
    time.sleep(2)

    arm2.open_gripper(70)
    time.sleep(2)
    U.move(arm2, v0[0], v0[1])


    sys.exit()

    if len(sys.argv) < 2:
        sys.exit()
    arg = sys.argv[1]

    if arg == 'throw':
        print(U.pos_rot_arm(arm1, nparrays=True))
        if len(sys.argv) > 2:
            arm1.open_gripper(70)
            time.sleep(2)
            arm1.open_gripper(-10)
            time.sleep(2)

        PSM1.move_to_pose(x7[0], x7[1])
        PSM1.move_to_pose(x11[0], x11[1])
        PSM1.move_to_pose(x9[0], x9[1])
        PSM1.move_to_pose(x12[0], x12[1])

    if arg == 'catch':
        psm2 = DVRKWrapper("PSM2")
        print psm2.pose
        psm2.move_to_pose(y0[0], y0[1])
        psm2.open_gripper(70)
        time.sleep(2)
        psm2.move_to_pose(y1[0], y1[1])
        psm2.open_gripper(-10)
        time.sleep(2)

    if arg == 'release':
        PSM1 = DVRKWrapper("PSM1")
        print PSM1.pose
        PSM1.open_gripper(70)
        time.sleep(2)
        PSM1.move_to_pose(z0[0], z0[1])
        PSM1.move_to_pose(z1[0], z1[1])

    if arg == 'handoff1':
        psm2 = DVRKWrapper("PSM2")
        print psm2.pose
        psm2.move_to_pose(w0[0], w0[1])
        time.sleep(1)
        psm2.move_to_pose(w1[0], w1[1])
        # psm2.open_gripper(70)
        # time.sleep(2)
        # psm2.open_gripper(-10)
        # time.sleep(2)

    if arg == 'handoff2':
        psm1 = DVRKWrapper("PSM1")
        print psm1.pose
        # psm1.move_to_pose(u0[0], u0[1])
        psm1.open_gripper(-10)
        time.sleep(2)

    if arg == 'handoff3':
        psm2 = DVRKWrapper("PSM2")
        print psm2.pose
        psm2.open_gripper(70)
        time.sleep(2)
        psm2.move_to_pose(v0[0], v0[1])

    if arg == 'psm2':
        psm2 = DVRKWrapper("PSM2")
        print psm2.pose

    if arg == 'test':
        psm2 = DVRKWrapper("PSM2")
        print psm2.pose
        psm2.translation((0,0,0.01))

    if arg == 'open':
        PSM1 = DVRKWrapper("PSM1")
        PSM1.open_gripper(70)
        time.sleep(2)
        PSM1.open_gripper(-15)
