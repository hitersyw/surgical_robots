"""
(c) January 2017 by Daniel Seita

Performs the second step in the GitHub documentation. This takes in the data
that I took as screenshots manually, and must save the following numpy arrays:

- X_train and y_train
- X_valid and y_valid
- X_test and y_test

in the `final_data` directory.

I'm not sure if we need validation right now but it doens't hurt to have one. If
we don't want it, then combine the train and valid stuff in the other scripts I
use before testing.

To make the required data, it basically does three major steps:

- Goes through images and eliminates any which have unusual size (to correct for
  potential human errors)
- Resize all images to be 32x32 (we might be able to do larger, IDK, but this
  lets us use standard CIFAR-10 style networks to minimize experimentation).
- Then split into train, valid, test and zero-mean it. This must be done before
  data augmentation, though we do the augmentation in the CNN code.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def sanity_checks(data_dirs):
    """ Run this to double-check data w/out changing anything. Then manually
    inspect those troublesome cases and delete if needed. """
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


def load_and_save(train_data, test_data, ratios, height=32, width=32):
    """ Now actually load the data into the numpy arrays. It's a bit sloppy
    since I modified this on short notice to handle an entirely new held-out
    testing set. Also, remember that normal probabilities come FIRST, then
    deformed probabilities.
    
    Specifically:
        - Load the images using cv2 in grayscale.
        - Resize them to (height,width) using linear interpolation.
        - Split based on train/valid/test, according to 'ratios' parameter,
          though the test data is now distinct so the ratios here only refers to
          the train and validation split.
        - Find the mean of the TRAINING data's statistics. Then normalize the
          three batches of data according to the TRAINING data's statistics.
          This should be the standard way to normalize.
        - Then save into numpy arrays.

    Note: detection of deformed/non-deformed happens with deformed in the
    directory name.
    """
    assert (len(ratios) == 2) and (np.sum(ratios) == 1)
    deform_yes_train = []
    deform_no_train = [] 
    deform_yes_test = []
    deform_no_test = [] 

    # Get things loaded, first training (and validation):
    for directory in train_data:
        for im in os.listdir(directory):
            if ('DS_Store' in im or '.txt' in im): continue
            image = cv2.imread(directory+'/'+im, cv2.IMREAD_GRAYSCALE)
            image_resized = cv2.resize(image, 
                                       (height,width), 
                                       interpolation=cv2.INTER_LINEAR)
            if ('deformed' in directory): 
                deform_yes_train.append(image_resized)
            else:
                deform_no_train.append(image_resized)

    # Now testing:
    for directory in test_data:
        for im in os.listdir(directory):
            if ('DS_Store' in im or '.txt' in im): continue
            image = cv2.imread(directory+'/'+im, cv2.IMREAD_GRAYSCALE)
            image_resized = cv2.resize(image, 
                                       (height,width), 
                                       interpolation=cv2.INTER_LINEAR)
            if ('deformed' in directory): 
                deform_yes_test.append(image_resized)
            else:
                deform_no_test.append(image_resized)

    # Balance the data and inspect sizes.
    len_y_tr, len_n_tr = len(deform_yes_train), len(deform_no_train)
    len_y_te, len_n_te = len(deform_yes_test), len(deform_no_test)
    deform_yes_train = np.array(deform_yes_train)
    deform_no_train = np.array(deform_no_train)
    deform_yes_test = np.array(deform_yes_test)
    deform_no_test = np.array(deform_no_test)

    print("Resized data loaded.")
    print("\tTrain (+Valid) Deformed: {}\n\tTrain Normal: {}".format(
            deform_yes_train.shape, deform_no_train.shape))
    print("\tTest Deformed: {}\n\tTest Normal: {}".format(
            deform_yes_test.shape, deform_no_test.shape))

    indices_yes_tr = np.random.choice(len_y_tr, min(len_y_tr, len_n_tr), replace=False)
    indices_no_tr = np.random.choice(len_n_tr, min(len_y_tr, len_n_tr), replace=False)
    deform_yes_train = deform_yes_train[indices_yes_tr]
    deform_no_train = deform_no_train[indices_no_tr]

    indices_yes_te = np.random.choice(len_y_te, min(len_y_te, len_n_te), replace=False)
    indices_no_te = np.random.choice(len_n_te, min(len_y_te, len_n_te), replace=False)
    deform_yes_test = deform_yes_test[indices_yes_te]
    deform_no_test = deform_no_test[indices_no_te]

    print("With balanced data now.")
    print("\tTrain (+Valid) Deformed: {}\n\tTrain Normal: {}".format(
            deform_yes_train.shape, deform_no_train.shape))
    print("\tTest (+Valid) Deformed: {}\n\tTest Normal: {}".format(
            deform_yes_test.shape, deform_no_test.shape))

    # Combine into one dataset and create labels: 0=NORMAL, 1=DEFORMED. 
    all_train_data = np.concatenate((deform_yes_train, deform_no_train))
    all_test_data = np.concatenate((deform_yes_test, deform_no_test))
    all_train_labels = np.concatenate((np.ones(deform_yes_train.shape[0]),
                                       np.zeros(deform_no_train.shape[0])))
    all_test_labels = np.concatenate((np.ones(deform_yes_test.shape[0]),
                                      np.zeros(deform_no_test.shape[0])))
    print("Train data together now.\n\tData: {}\n\tLabels: {}".format(
            all_train_data.shape, all_train_labels.shape))
    print("Test data together now.\n\tData: {}\n\tLabels: {}".format(
            all_test_data.shape, all_test_labels.shape))

    # Then shuffle & split. Use the same indices to keep data & labels matched.
    N = all_train_data.shape[0]
    indices = np.random.permutation(N)
    indices_train = indices[ : int(N*ratios[0])]
    indices_valid = indices[int(N*ratios[0]) : ]

    X_train = all_train_data[indices_train].astype('float32')
    y_train = all_train_labels[indices_train]
    X_valid = all_train_data[indices_valid].astype('float32')
    y_valid = all_train_labels[indices_valid]

    X_test = all_test_data.astype('float32')
    y_test = all_test_labels

    print("X_train {}, y_train {}".format(X_train.shape, y_train.shape))
    print("X_valid {}, y_valid {}".format(X_valid.shape, y_valid.shape))
    print("X_test {}, y_test {}".format(X_test.shape, y_test.shape))

    # Now center the images, and then save. Whew. SAVE THE MEAN!!
    mean_image = np.mean(X_train, axis=0).astype('float32')
    X_train -= mean_image
    X_valid -= mean_image
    X_test -= mean_image
    print("Saving the mean (shape = {}) ...".format(mean_image.shape))
    np.save("final_data/X_train_MEAN", mean_image)

    np.save("final_data/X_train", X_train)
    np.save("final_data/y_train", y_train)
    np.save("final_data/X_valid", X_valid)
    np.save("final_data/y_valid", y_valid)
    np.save("final_data/X_test", X_test)
    np.save("final_data/y_test", y_test)

    # Some last-minute sanity checks.
    print("min, max X_train = {}, {}".format(np.min(X_train), np.max(X_train)))
    print("min, max X_valid = {}, {}".format(np.min(X_valid), np.max(X_valid)))
    print("min, max X_test = {}, {}".format(np.min(X_test), np.max(X_test)))


if __name__ == "__main__":
    data_dirs_train = ['data_raw_train/im_left_deformed',
                       'data_raw_train/im_left_deformed_hard',
                       'data_raw_train/im_left_normal',
                       'data_raw_train/im_left_normal_hard']
                       # 'data_raw_train/im_right_deformed',
                       # 'data_raw_train/im_right_normal'
    data_dirs_test = ['data_raw_test/im_left_deformed',
                      'data_raw_test/im_left_deformed_hard',
                      'data_raw_test/im_left_normal',
                      'data_raw_test/im_left_normal_hard']
                      # 'data_raw_test/im_right_deformed',
                      # 'data_raw_test/im_right_normal'
    # once the data is clean, I don't need to run this method any more.
    sanity_checks(data_dirs_train)
    sanity_checks(data_dirs_test)

    height, width = 32, 32
    train_valid_ratio = [0.90, 0.10]
    load_and_save(train_data=data_dirs_train, 
                  test_data=data_dirs_test, 
                  ratios=train_valid_ratio,
                  height=height, 
                  width=width)
    print("All done!")
