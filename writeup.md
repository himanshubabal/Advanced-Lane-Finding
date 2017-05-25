## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./test_images/undistort_output.png "Undistorted"
[image2]: ./test_images/vlc_42.jpg "Original Image"
[image3]: ./test_images/undistorted_img.jpg "Distortion Free Image"
[image4]: ./test_images/color_binary.jpg "Gradient"
[image5]: ./test_images/pers.jpg "Perspective Transform"
[image6]: ./test_images/smooth_window.jpg "Smooth Window"
[image7]: ./test_images/final.jpg "Final output"
[image8]: ./test_images/window.jpg "Sliding Window"
[image9]: ./test_images/pixel.jpg "Lane Pixels"
[video1]: ./project_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file `cam_calibrate.py` located in project directory.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

Original Image            |  Distortion Free image
:-------------------------:|:-------------------------:
![alt text][image2]  |  ![alt text][image3]


Code can be fount in file `undistort_image.py`. I get the camera matrix from pickle saved in `cam_calibrate.py` and use opencv's `cv2.undistort()` method to undistort the image


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #10 through #39 in `colour_gradient.py`).  Here's an example of my output for this step.

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `get_perspectiveTransform()`, which appears in lines 14 through 28 in the file `perspective_trans.py`.  The `get_perspectiveTransform()` function takes as inputs an image (`img`), and I have defined `src` and `dest` points.  I chose to hardcode the points as follows:

```python
p1, p2, p3, p4 = [585, 450], [695, 450], [200, 720], [1120, 720]
d1, d2, d3, d4 = [300, 0], [1000, 0], [300, 720], [1000, 720]
```

After many trials with adjusting the numbers, I found these to be optimal for the purpose.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code to fit polynomial can be found in `lane_lines.py`.
I will be explaining methods in this file and how each method works

###### `get_lane_lines_window(binary_warped_image)`

This method takes `binary_warped_image` as input (image we obtain after applying camera undistortion, color gradient and perspective transform).
Then it uses histogram to figure out where in the image are lanes located. It then divides region in 2 parts and scans both parts with sliding window techniques to find lane markers. It results into something like this -

![alt text][image8]

It also returns `left_fit` and `right_fit`. Reason explained in next method


###### `get_lane_lines_using_prev_info()`

It takes `left_fit` and `right_fit` along with image as inputs. It is used to save time. As we already know the region in which lanes in earlier frame lied, we can use that info to scan only that area for lanes. It will save us some time. Process is quite similar to above described.
It results into -

![alt text][image6]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 212 through 237 in the method `get_curvature()` my code in `lane_lines.py`

```python
# Concversion from pixel values to meters
ym_per_pix = 30.0 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

# Calculating curvature near the car as curvature
# varies from near the car to away from car
y_eval = np.max(ploty)

# Then we fit a polynomial on obtained x and y values
# for both parts. X values of curve can be obtained
# from previous methods
left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
# similarly get right curve

# Calculating distance from center
screen_middel_pixel = img.shape[1] / 2
left_lane_pixel = np.min(left_fitx)    # x position for left lane
right_lane_pixel = np.max(right_fitx)   # x position for right lane
car_middle_pixel = int((right_lane_pixel + left_lane_pixel) / 2)
screen_off_center = screen_middel_pixel - car_middle_pixel
meters_off_center = xm_per_pix * screen_off_center  
```


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.


I implemented this step in 7th cell in Ipython noteboo in `pipeline.ipynb` in the function `process_final()`.  Here is an example of my result on a test image:


![alt text][image7]




---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Video can be found in project directory with the name `project_video_out.mp4`




---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Initially, I had problems with too much jittering in the video, but then I decided to take an average of the filled polynomial over last 25 frames.

Since the video is 50 seconds long and there are nearly 1200 frames, so that makes it 25 frames/sec.
This I believe that by averaging it over last 25 frames, resulting output would depend on last 1 second and thus will not jump randomly.

Turns out it works perfectly.
