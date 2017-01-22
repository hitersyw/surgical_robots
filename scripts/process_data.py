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
  lets us use standard CIFAR-10 networks to minimize experimentation).
- Then split into train, valid, test AND then do data augmentation. This must be
  done _after_ the split!
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys


if __name__ == "__main__":
    pass
