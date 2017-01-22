"""
(c) January 2017 by Daniel Seita

Performs the second step in the GitHub documentation. This takes in the data
that I took as screenshots manually, and must save the following numpy arrays:

- X_train and y_train
- X_valid and y_valid
- X_test and y_test

I'm not sure if we need validation right now but it doens't hurt to have one. If
we don't want it, then combine the train and valid stuff in the other scripts I
use before testing.

To make the required data, it basically does three major steps:

- Goes through images and eliminates any which have unusual size (to correct for
  potential human errors)
- Resize all images to be 32x32 (we might be able to do larger, IDK, but this
  lets us use standard CIFAR-10 style networks to minimize experimentation).
- Then split into train, valid, test. This must be done before data
  augmentation, though we do the augmentation in the CNN code.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import scipy
from scipy import misc


def sanity_checks(data_dirs):
    """ Run this to double-check data w/out changing anything. Then manually
    inspect those troublesome cases and delete if needed. This is mainly for a
    sanity check.
    """
    shapes = {}

    # Get shapes loaded.
    for directory in data_dirs:
        shapes[directory] = []
        for im in os.listdir(directory):
            if ('DS_Store' in im or '.txt' in im): continue
            image = cv2.imread(directory+'/'+im, cv2.IMREAD_GRAYSCALE)
            shapes[directory].append(image.shape)

    for key in shapes:
        # Now inspect first the raw shapes
        shapes[key] = np.array(shapes[key])
        ss = shapes[key]
        print("\n\nKEY = {} with {} images:".format(key, ss.shape[0]))
        print("\tmean {}\n\tmin {}\n\tmax {}\n\tstd {}".format(
                    np.mean(ss, axis=0), 
                    np.min(ss, axis=0),
                    np.max(ss, axis=0),
                    np.std(ss, axis=0))
        )

        # Then the shape ratios (first_dim / second_dim)
        dim_ratio = ss[:,0].astype('float32') / ss[:,1]
        print("\nnow dimension ratios:")
        print("\tmean {:.4f}\n\tmin {:.4f}\n\tmax {:.4f}\n\tstd {:.4f}".format(
                    np.mean(dim_ratio, axis=0), 
                    np.min(dim_ratio, axis=0),
                    np.max(dim_ratio, axis=0),
                    np.std(dim_ratio, axis=0))
        )

        # Finally, if desired, use this to explicitly detect images. It's
        # easiest to find these images if you do `ls -lh > test.txt` and open
        # the file in vim, which numbers them by line.
        print("\nspecial images to detect:")
        print("large height: {}".format(np.where(ss[:,0] > 600)[0]))
        print("large width: {}".format(np.where(ss[:,1] > 600)[0]))
        print("high ratio: {}".format(np.where(dim_ratio > 1.4)[0]))
        print("low ratio: {}".format(np.where(dim_ratio < 0.6)[0]))


def load_and_save(data_dirs):
    """ Now actually load the data into the numpy arrays. Also do preprocessing
    here to zero-mean things, etc. We will _not_ do data augmentation here. 
    """
    pass


if __name__ == "__main__":
    data_dirs = ['im_left_deformed',
                 'im_right_deformed',
                 'im_left_normal',
                 'im_right_normal']
    #sanity_checks(data_dirs)
    load_and_save(data_dirs)
