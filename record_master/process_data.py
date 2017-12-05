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
np.set_printoptions(suppress=True, edgeitems=10, linewidth=180, precision=5)

# For HSV.
LOWER  = np.array([0,0,0])
HIGHER = np.array([255,50,255])

# Other stuff
DIM = 256
TRAIN_FRAC = 0.8


def load_pickle_to_list(filename, squeeze=True):
    """
    I'm putting this here because if we do this on another computer (e.g., the
    tritons), I can't import rospy (from the utils.py file).
    """
    f = open(filename,'r')
    data = []
    while True:
        try:
            data.append(pickle.load(f))
        except EOFError:
            break
    assert len(data) >= 1
    f.close()

    # if length is one, might as well just 'squeeze' it...
    if len(data) == 1 and squeeze:
        return data[0]
    else:
        return data


def process_one_image(img):
    """ Important! Processes one image (stored as a numpy array).

    The dvrk needs to do the same thing when it is in action. Unfortunately, I
    think there will be issues with timing ... so try and keep the
    pre-processing to a minimum. And we also have to pass it through a neural
    network ... uh oh.

    TODO: test this out again once we have better paint for needle detection.
    """
    # Cropping boundaries. Requires some tweaking but I _think_ these work.
    x,y,w,h = 600, 0, 1024, 1024
    #cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,0), 2) # Debug bounding boxes.
    img = img[y:y+h, x:x+w]

    # Resize. Do it here to make the rest of the process slightly faster.
    scale = 0.25
    img = cv2.resize(img, (int(w*scale),int(h*scale)), interpolation=cv2.INTER_LINEAR)

    # HSV Mask. This works well for the non-painted needles.
    img  = cv2.medianBlur(img, 9)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER, HIGHER)
    img  = cv2.bitwise_and(img, img, mask=mask)

    # Grayscale.
    img = cv2.cvtColor( cv2.cvtColor(img, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)

    return img


def process_images(demo, left):
    """ Iterates through directories to process images. """
    if left:
        new_path = 'data/'+demo+'/left_proc/'
        head ='data/'+demo+'/left_endoscope/'
    else:
        new_path = 'data/'+demo+'/right_proc/'
        head ='data/'+demo+'/right_endoscope/'

    if not os.path.exists(new_path):
        os.mkdir(new_path)
    img_names = sorted(os.listdir(head))

    for name in img_names:
        img = cv2.imread(head+name)

        # Crop and then downsample. The dvrk must do the same thing!!
        img = process_one_image(img)

        # Save images.
        new_name = name[:-4] +'_proc'+ name[-4:]
        cv2.imwrite(new_path+new_name, res)
    print("Done with image processing.")


def form_data(dirs):
    """ Form the actual dataset.

    Return a dictionary that has X_train, y_train, X_valid, and y_valid keys.
    """
    # One item in these lists is an np.array for states/actions in one demo.
    all_stats = []
    all_actions = []

    for demo in dirs:
        dpath = 'data/'+demo
        proc_path_left  = dpath+'/left_proc/'
        proc_path_right = dpath+'/right_proc/'
        img_names_left  = sorted(os.listdir(proc_path_left))
        img_names_right = sorted(os.listdir(proc_path_right))

        # Load limits and stats (which is a _dictionary_).
        limits = np.loadtxt(dpath+'/limits.txt')
        assert len(limits) == 2
        lower, upper = int(limits[0]), int(limits[1])
        stats = load_pickle_to_list(dpath+'/demo_stats.p')

        # Finally, iterate through left/right images and form the data.
        states = []
        actions = []

        for idx,(name_l,name_r) in enumerate(zip(img_names_left,img_names_right)):
            if not (lower <= idx <= upper):
                continue
            img_l = cv2.imread(proc_path_left + name_l)
            img_r = cv2.imread(proc_path_right + name_r)
            assert img_l.shape == (DIM,DIM) and img_r.shape == (DIM,DIM)

            # Up to debate ... I'm putting them in two channels.
            img_l = np.expand_dims(img_l, 0)
            img_r = np.expand_dims(img_r, 0)
            imgs  = np.concatenate((img_l, img_r), axis=0)
            assert imgs.shape == (2,DIM,DIM)

            # Also up to debate, on how I extract the actions.
            pos_rot    = stats['pos_rot_1']
            current    = np.array(pos_rot[idx])
            subsequent = np.array(pos_rot[idx+1])
            action     = subsequent - current

            states.append(imgs)
            actions.append(action)

        # Append to our two list of lists. Making these np.arrays should result
        # in shapes of (T, 2, DIM, DIM) where T is the length of the demo.
        all_states.append( np.array(states) )
        all_actions.append( np.array(actions) )

    # Collect data in numpy form. We might want to keep track of our data in
    # un-shuffled form, though ...
    X_data = np.concatenate(all_states)
    y_data = np.concatenate(all_actions)
    N = X_data.shape[0]
    indices = np.random.permutation(N)
    cutoff = int(TRAIN_FRAC * N)
    idx_train = indices[:cutoff]
    idx_valid = indices[cutoff:]
    print("\n(Forming the train/valid split now ...)")
    print("X_data.shape: {}".format(X_data.shape))
    print("y_data.shape: {}".format(y_data.shape))
    print("N = {}, cutoff (for training) = {}".format(N, cutoff))

    data = {}
    data['X_train'] = X_data[ : idx_train ]
    data['y_train'] = y_data[ : idx_train ]
    data['X_valid'] = X_data[ idx_valid : ]
    data['y_valid'] = y_data[ idx_valid : ]
    return data


def sanity_checks_and_save(data):
    """ Double check our data and save it. """
    print("\nNow let's double check our data to ensure that it makes sense.")
    print("X_train.shape: {}".format(data['X_train'].shape))
    print("y_train.shape: {}".format(data['y_train'].shape))
    print("X_valid.shape: {}".format(data['X_valid'].shape))
    print("y_valid.shape: {}".format(data['y_valid'].shape))
    pass


if __name__ == "__main__":
    dirs = sorted([x for x in os.listdir('data') if 'demo_' in x])
    for d in dirs:
        assert os.path.exists('data/'+d+'/limits.txt')
        assert os.path.exists('data/'+d+'/demo_stats.p')
    print("Processing data based on {} demonstrations.".format(len(dirs)))

    # Process images if necessary.
    for idx,d in enumerate(dirs):
        print("\nHere is demo/directory index {}".format(idx))
        process_images(demo=d, left=True)
        process_images(demo=d, left=False)

    # Form training and validation data. Then save.
    data = form_data(dirs)
    sanity_checks_and_save(data)
