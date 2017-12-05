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


def process_actions():
    """ Get actions.
    
    I think these will be the difference between the pos/rot vectors, but I'll
    have to see.
    """ 
    pass


if __name__ == "__main__":
    dirs = sorted([x for x in os.listdir('data') if 'demo_' in x])
    for d in dirs:
        assert os.path.exists('data/'+d+'/limits.txt')
        assert os.path.exists('data/'+d+'/demo_stats.p')
    print("Processing data based on {} demonstrations.".format(len(dirs)))

    print("debugging only, using first demonstration")
    process_images(demo=dirs[0])
    process_actions()
