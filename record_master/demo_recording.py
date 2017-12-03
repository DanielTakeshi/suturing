"""
Based on Carolyn's original code.
See README for how this should work.
"""
import Tkinter
import pickle, time, os, scipy, sys
from multiprocessing import Process
import numpy as np
import IPython
# from ImageSubscriber import ImageSubscriber
import matplotlib.pyplot as plt
import rospy
import multiprocessing
sys.path.append('../')
from dvrk.robot import robot
from autolab.data_collector import DataCollector
from collections import defaultdict
from scripts import utils as U

# Check these ... fine tune the interval. Using 1.0 seems too coarse, but using
# 0.33 means I often see images that are exactly the same across different
# times. Somewhere in between is probably good.
INTERVAL = 0.5
USE_PSM2 = False


def startCallback():
    """
    Once start is pressed, this runs. Have one process dedicated to recording
    the poses.
    """
    global record_process, f1, f2, exit
    print("start")
    if record_process != None:
        print " You are already recording"
        return
    if exit.is_set():
        exit.clear()
    record_process = Process(target=start_listening, args=(exit,))
    record_process.start()


def stopCallback():
    global record_process
    print "stop"
    if record_process == None:
        print " Nothing currently recording"
        return
    exit.set()
    record_process.terminate()
    record_process.join()
    record_process = None


def exitCallback():
    global record_process, f1, f2
    print "exit"
    if record_process != None:
        print " terminating process"
        if(exit.is_set()):
            pass
        else:
            exit.set()
        record_process.terminate()
        record_process.join()
    top.destroy()
   
    if USE_PSM2:
        if f1 != None and f2 !=None:
            f1.close()
            f2.close()
    else:
        if f1 != None:
            f1.close()
    raise SystemExit()


def start_listening(exit):
    """ Records information from one trajectory of a human adjusting the tools.

    Save the trajectory at the end, since saving the images as we go is very
    costly. Images are extracted from the autolab data collector.

    The `INTERVAL` variable is important: determines how precise we want to be
    with timing. I think 1 second is an upper bound, maybe 0.5 is good.
    """
    pos1, pos2 = None, None
    grip1, grip2 = None, None
    directory = E.get()
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.makedirs(directory+ "/left_endoscope")
        os.makedirs(directory+ "/right_endoscope")

    d = DataCollector()
    psm1 = robot("PSM1")
    if USE_PSM2:
        psm2 = robot("PSM2")
    time.sleep(1)
    count = 0
    start_t = time.time()

    # Add data here, then save these three things.
    stats = defaultdict(list)
    im_left_raw = []
    im_right_raw = []

    while not exit.is_set():
        t = rospy.get_rostime()
        t = t.secs + t.nsecs/1e9
        current_t = (time.time() - start_t)

        pose1 = psm1.get_current_cartesian_position()
        pos1, rot1 = U.pos_rot_cpos(pose1)
        joint1 = psm1.get_current_joint_position()
        grip1 = [joint1[-1] * 180 / np.pi]
        print(current_t, pose1, t)

        if USE_PSM2:
            pose2 = psm2.get_current_cartesian_position()
            pos2, rot2 = U.pos_rot_cpos(pose2)
            joint2 = psm2.get_current_joint_position()
            grip2 = [joint2[-1] * 180 / np.pi]

        stats['time_ros'].append(t)
        stats['time_secs'].append(current_t)
        stats['pos_rot_1'].append( pos1+rot1 )
        stats['g_joint_1'].append( list(grip1)+list(joint1) )

        if USE_PSM2:
            stats['pos_rot_2'].append( pos2+rot2 )
            stats['g_joint_2'].append( list(grip2)+list(joint2) )

        im_left_raw.append(d.left['raw'])
        im_right_raw.append(d.right['raw'])

        count += 1
        time.sleep(INTERVAL)

    limit = 1000 
    if count > limit:
        print("count {} > limit {} too much data".format(count, limit))
        sys.exit()

    print("saving images and stats dict (count: {}), may take a few minutes".format(count))
    num_digits = len(str(abs(count)))
    for c in range(count):
        if c % 5 == 0:
            print("saving index {} now ...".format(c))
        idx = str(c).zfill(num_digits)
        left  = directory+ "/left_endoscope/im_raw_left_" +idx+ ".png"
        right = directory+ "/right_endoscope/im_raw_right_" +idx+ ".png"
        scipy.misc.imsave(left,  im_left_raw[c])
        scipy.misc.imsave(right, im_right_raw[c])
    print("DONE with saving images")
    U.store_pickle(fname=directory+"/demo_stats.p", info=stats)


if __name__ == '__main__':
    # f1 and f2 are the file objects for saving left/right poses.
    # record_process records the positions of the robot.

    exit = multiprocessing.Event()
    f1, f2 = None, None
    record_process = None

    num_dirs = len([x for x in os.listdir('data/') if 'demo' in x])
    directory = "data/demo_" +str(num_dirs).zfill(3)

    # Getting GUI up, but I should be fine with the default directory.
    # When buttons B, C, D are pressed, they result in the callback method.
    # Press B, then do stuff, then press C and then D. Later, combine C and D.
    top = Tkinter.Tk()
    top.title('Pose Listener')
    top.geometry('400x200')
    B = Tkinter.Button(top, text="Start Recording", command=startCallback)
    C = Tkinter.Button(top, text="Stop Recording", command=stopCallback)
    D = Tkinter.Button(top, text="Exit", command=exitCallback)
    E = Tkinter.Entry(top)
    B.pack()
    C.pack()
    D.pack()
    E.pack()
    E.delete(0, Tkinter.END)
    E.insert(0, directory)

    # Seems to be how Tkinter is used; have an infinite loop here.
    top.mainloop()
