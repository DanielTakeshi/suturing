# Demo Recording from Master

Here's how this should work.

**Setup checklist** (make sure I read this each time!!):

- Put two layers of the same green foam (with the X mark), put in same setup.
  Make sure they have rails so that they won't leave the setup (as the gripper
  can grip them as well ...). Maybe tape is better? And put tape in between the
  two foams.

- Put same needle (with the wire glued on it) in "roughly" the same location. I
  think we want some similar stuff but some diversity is good. The location will
  need to be tuned a bit because not all initial configurations can lead to it
  reaching the target position.

- The location should be such that no further downward movement is needed to
  grip the needle.

- Speaking of the target, we're going to need that in the target image, so that
  I (the human) can see what the objective is going to be. So ... I'll have to
  figure that out somehow. The issue is that that's going to be in the images,
  so we'll have to ensure that the behavioral cloning doesn't try and track
  that.

- **TODO** paint the needles in some way. I know we'll have to do some image
  processing, however we should definitely be saving the raw images. The
  needles, I think, should be painted yellow uniformly, but then have some color
  for the tip and some other color for the other end? That way I can do the
  processing that I wanted, but also perhaps we can extract an image of the full
  needle.

Record the above with camera images and ensure that it remains consistent. 

**Usage**:

- Run the demo collection python command. A pop-up window appears.

- Before clicking start, use the master clutch so that the arms are in a
  reasonable spot. This is to prevent any clutches while we're in progress.

- Try to smoothly adjust the dvrk so that it smoothly reaches its target. Easier
  said than done, I know ... **TODO need to train and get camera set up**.

- Double check images immediately after each trajectory. Delete the directory if
  there was a failure.

- **TODO**: don't save the first and last 5 since those are typically nothing
  due to lag between when we start and finish ... 

**Later**:

- With this data, we have `(images,position)` and will (a) process images in
  some unforseen way, and (b) the action should be the position delta,
  `pos_{t+1}-pos_{t}` right? That way we can take that and add it to the current
  position in practice.

- **TODO** once I practice, hopefully the needles start going in the right spot.
  Then we practice on the real surgical phantom, but with the **same** painting
  setup for the needles.
