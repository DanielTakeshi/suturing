"""
Process data from the demonstrations.  Run this script in this directory. No
command line arguments needed.

(c) 2017 by Daniel Seita
"""
import argparse
import cv2
import numpy as np
import os
import pickle
import sys
sys.path.append('../')
from scripts import utils as U
np.set_printoptions(suppress=True, edgeitems=10, linewidth=180, precision=5)


def process_images(demo):
    """ Processes the raw images.
    
    Techniques: bounding box to crop, and then use Sanjay's image extraction
    stuff. TODO: test this out again once we have better paint for needle
    detection.
    """
    new_path = 'data/'+demo+'/left_proc/'
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    head ='data/'+demo+'/left_endoscope/'
    img_names = sorted(os.listdir(head))

    # Cropping boundaries. Requires some tweaking but I _think_ these work.
    x,y,w,h = 650, 0, 1024, 1024

    for name in img_names:
        img = cv2.imread(head+name)

        # Crop
        #cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,0), 2)
        img = img[y:y+h, x:x+w]

        # HSV Mask. This works well for the non-painted needles.
        img = cv2.medianBlur(img, 9)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0,0,0])
        higher = np.array([255,50,255])
        mask = cv2.inRange(hsv, lower, higher)
        res = cv2.bitwise_and(img, img, mask=mask)
                                        
        # Save images.
        new_name = name[:-4] +'_proc'+ name[-4:]
        #cv2.imwrite(new_path+name, img)
        cv2.imwrite(new_path+new_name, res)

    print("Done with image processing.")


def process_actions(demo):
    """ Get actions.
    
    I think these will be the difference between the pos/rot vectors, but I'll
    have to see. Let's test this out before scaling up further.
    """ 
    dpath = 'data/'+demo
    new_path = dpath+'/left_proc/'
    assert os.path.exists(new_path) # made in previous method
    img_names = sorted(os.listdir(new_path))

    # Load limits and stats (which is a _dictionary_).
    limits = np.loadtxt(dpath+'/limits.txt')
    assert len(limits) == 2
    lower, upper = int(limits[0]), int(limits[1])
    stats = U.load_pickle_to_list(dpath+'/demo_stats.p')

    for idx,img_name in enumerate(img_names):
        if not (lower <= idx <= upper):
            continue
        img = cv2.imread(new_path+img_name)
        pos_rot = stats['pos_rot_1']

        current    = np.array(pos_rot[idx])
        subsequent = np.array(pos_rot[idx+1])

        print("\nimg {}".format(new_path+img_name))
        #print("pos_rot: {}".format(pos_rot[idx]))
        print("delta: {}".format(subsequent - current))


if __name__ == "__main__":
    dirs = sorted([x for x in os.listdir('data') if 'demo_' in x])
    for d in dirs:
        assert os.path.exists('data/'+d+'/limits.txt')
        assert os.path.exists('data/'+d+'/demo_stats.p')
    print("Processing data based on {} demonstrations.".format(len(dirs)))

    print("debugging only, using one demonstration, I'd do a for loop otherwise ...")
    #process_images(demo=dirs[0])
    process_actions(demo=dirs[0])
