# surgical_robots

Here are the steps I did, just to be clear:

- Directories `camera_left` and `camera_right` contain the raw images from the
  da vinci's two cameras. There are 67 for each arm, hence 134 total.

- Then on my local computer, I just took a bunch of screenshots (CMD-SHIFT-4) to
  get a huge list of images. I took screenshots of all the **deformed** stuff,
  basically bounding boxes. I then put those in `im_left_deformed` or
  `im_right_deformed`, depending on whether they came from the left or right. I
  repeated the process for the **non-deformed** stuff, where I took care to try
  and pick a diversity of non-deformed stuff, but to also ensure minimal to no
  overlap with a deformed image. (The distinction between left/right shouldn't
  matter but I'll keep track of them anyway in case the lighting difference
  causes too much confusion, but I think it will be beneficial to have it.)
  Those "normal" images are stored in `im_left_normal` and `im_right_normal`.

  Note I: I tried to be consistent and also to avoid noise, i.e. the border of
  the gauze.

  Note II: Be sure to bound stuff from top to bottom so I don't forget any.

  Note III: If there are images I made which are particularly small, I should
  eliminate them.  On the other hand, even the smallest ones are larger than
  28x28, I think. How small do we want them? 32x32?
  
  Note IV: Or really wide. I tried not to make really small or really wide ones
  ... but it's tricky, how do we usually do this with variable patch sizes? That
  sounds like a job for attention models but I'm not sure if we need anything
  sophisticated like that. Yeah, I think if there's something really wide, I
  should just split it in two and make two separate images. Hey, it makes our
  dataset larger! EDIT: Programmatically check the images for sizes. That's
  easier and faster.

  Note V: I will rename these frames automatically using Python later.

- TODO preprocessing to clear out junk stuff? Pressing space in preview means
  arrow keys can be used to scroll quickly to check for junk cases.

- Then it's data augmentation. TODO

- Then train a CNN using keras and Tensorflow. TODO

