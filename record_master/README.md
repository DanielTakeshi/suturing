# Demo Recording from Master

First, make sure the setup is correct, before we begin collecting data. Second,
each time I run the demo recording code, it will record ONE trajectory. There's
a checklist for that.


## Setup

- Put three pieces of green foam on top of each other and put tape on them so
  they don't move up (well, usually...) when the gripper also grips the foam
  with the needle.

- Put the green foams inside pins screwed down on the workspace.

- For simplicity, do not even put a gripper in the PSM2. This frees my left hand
  to help manipulate the PSM1 when using the master tools.


## Demo Recording (One Trajectory)

**Read this each time!!**

- With the PSM1 in a place where it's not gripping a needle or touching the
  foam, click "home" in the teleop so that it "resets" to a good position.

  **Note**: perhaps this is not needed if there are no errors, and if the master
  tools can be adjusted using the foot clutch. Basically, if I'm at a good spot
  and teleop is on and there are no warnings, I hope I will be OK. Otherwise,
  use foot clutch to "reset".

- Put the needle in a place that corresponds roughly to the center of the
  endoscope cameras and in a place where it can see clearly, and put the PSM1
  gripper (with SNAP) so that it only has to grip the needle.

- If necessary, click "teleop". Check that there are no warnings with the joint
  angles. If there are, restart the process.

- Now run the script `demo_recording.py`. A pop-up window appears. Click start.

- Move the master tools. :-) BUT MOVE SLOWLY, as the camera images will update
  only every 0.5 seconds by default.

- When I'm done, click stop and wait a few minutes for it to save the data. OR
  if I know the trajectory was a failure, just kill the program and delete the
  `demo_XYZ` directory.

- Next, inside each of the `demo_XYZ` directories, in a file called `limits.txt`
  (and please use the exact name here) put in TWO numbers. The first is the time
  step at which we should begin considering the data. The second is the time
  step at which we should stop considering the data. This way, we ignore the
  first N images of no activity and the last M images of no activity. Put these
  as separate lines. That is, the entire `limits.txt` file should look like
  this:

  ```
  5
  18
  ```

  if, for instance, 5 and 18 are the appropriate values. These should be
  *inclusive* for both. There might be a way to do this automatically but we
  should always be checking the data anyway.

  **Food for thought**: is it actually better to include lots of the images at
  the end, when it's at the target? How will we be able to terminate the
  trajectory, after all? And it would be great to see it slow down near the end.

- Finally, use the hand clutch to release the needle and move the gripper to a
  good spot, and click "home".

  **UPDATE**: actually maybe not ... for now I think I can get away with using
  the *hand* clutch to move the gripper to where it will be gripping for the
  next demo, and then using the *foot* clutch to adjust the master tools so that
  they're not stuck at the top.

Then, repeat the above for additional trajectories! Once I have a dataset, I can
then do a bunch of other stuff ...

**Note**: some data cleaning may be needed since I see images that are
duplicates of each other ... but that can be handled at a later stage by
checking with `np.allclose`. EDIT: that's because of the subscriber rate.
