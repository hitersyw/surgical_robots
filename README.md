# Surgical Robots

**Update**: for preliminary results, see the `log_files` directory for the CNN's
output.

## Main Idea

This repository will do some classification on surgical gauze for deformed vs
normal testing. There are three major steps:

- Getting the data somehow
- Pre-processing it for a CNN
- Running a CNN

I'll explain each in turn.

## Getting the Data

- I took two pieces of surgical gauze and saved lots of images manually. I
  altered the images gradually, so as to get more variation in the images.
  Alterations include more deformations, rotations, pushing/pulling the gauze to
  make it flatter or slightly bumpy, slight stretches, and changing the light
  orientation.

- I put all these in the directories `camera_left` and `camera_right`. These
  contain the raw images from the da vinci's two cameras. There are 67 for each
  arm, hence 134 total. (The distinction between left/right shouldn't matter but
  I'll keep track of them anyway in case the lighting difference causes too much
  confusion, but I think it will be beneficial to have it.)

- Then on my local computer, I just took a bunch of screenshots (CMD-SHIFT-4) to
  get a huge list of images. I took screenshots of all the **deformed** stuff,
  basically bounding boxes. I then put those in `im_left_deformed` or
  `im_right_deformed`. I repeated this for the **non-deformed** stuff, where I
  took care to try and pick a diversity of non-deformed stuff, but to also
  ensure minimal to no overlap with a deformed image. These "normal" images are
  stored in `im_left_normal` and `im_right_normal`.

  - Note I: I tried to be consistent in the way I was taking screenshots, by
    taking similar sizes, avoiding the border of the gauze, not making them too
    large or wide since we have to resize them later, etc.
  - Note II: Use space bars in Mac preview. It makes sorting faster. =)

## Pre-Processing

Next, run `python scripts/process_data.py` from the home directory of the
repository. This will do the following main steps:

- Check all images for size and height/width sanity checks.

  - To be specific, I can detect the *indices* of the screenshots with unusual
    heights, widths, or height/width ratios. Then I can find them manually and
    get rid of them. I can find the index of files with the command `ls -lh >
    test.txt` and then inspect `test.txt` using vim with line indices (be sure
    to add one since it starts from zero).

- Resize images to be the same size, then shuffle into train/test.

- Normalize them by centering the mean about 0. (This is the standard way to
  deal with images; we actually don't often do variance normalization.)

- Then saves them into numpy arrays for later use.

## Running CNNs

Now this will use the training data from the last step to run a CNN classifier.
Simply run `python scripts/run_network.py`, again from the home directory of the
repository. It contains built-in code for data augmentation, and also code for
plotting.
