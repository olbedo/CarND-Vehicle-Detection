# Vehicle Detection and Tracking Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier

* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.

* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.

* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.

* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.

* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./pics/car_not_car.png
[image2]: ./pics/features.png
[image3]: ./pics/examples.png
[image4]: ./pics/boxes_heatmaps.png
[image5]: ./pics/heatmap_final_boxes.png
[video1]: ./project_video.mp4

## Rubric Points
**Here I will consider the rubric points individually and describe how I addressed each point in my implementation.**

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 123 through 181 of the file called `p5.py`.

I started by reading in all the `vehicle` and `non-vehicle` images (code lines 195 through 208). Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and settled on `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, because it was a good trade-off between accuracy and performance. The best results I got when I converted the images to `YUV` color space beforehand. I used all 3 channels in the final solution.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using all three features: HOG, spatial bins and color histogram (code lines 245 through 290). I also experimented with non-linear SVM, AdaBoost and Random Forests. The non-linear SVM algorithm showed very good performance in detecting the cars but was painfully slow, not a good choice for a real-time application like this. AdaBoost and Random Forests performed worse and were also slower than linear SVM. So I used a linear SVM in the final solution.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

For sliding window search I used a similar algorithm to the one provided in (code lines 324 through 396). I choose the relevant part of the image and scale it appropriately. Then I apply HOG in one go.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. I used the scales 0.75, 1.0, 1.5 and 2.0. As step width I used 2 pixels in each case. Other values resulted in too many or too few detections. Here are some example images:

![alt text][image3]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
I attached the video to the submission.

Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected (code lines 398 through 427).

The best results I got combining the detections over a series of 10 images and using a threshold of 70
Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video.

### Here are six frames and their corresponding heatmaps:

![alt text][image4]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames and the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image5]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

I experimented with different combinations of features: 1 or 3 channels of HOG features, with or without spatial binning and color histograms, in RGB and with conversions to HLS, YUV and YCrCb. I also tried different settings for the parameters applied to the feature generation. Finally, I used the all features with the parameters mentioned above.

For the classification I played around with linear and non-linear SVM, Adaboost and Random Forests. The best results regarding detection performance and speed provided the linear SVM algorithm. When I got nice results, I created a pipeline to do the detection with different scales. It was difficult to find a good set of scales and regions of interest for these scales. I tried to focus on the difficult parts, namly the areas with shadows. When I thought I got a good set of values and applied it on the project video I realized that it was not ideal for other parts, like the white car. So I had to experiment a lot to finally get to a solution that does not result in too many false positive detections in the dark areas while not failing to detect the white car in other sequences.

I think the most critical part is the classifier. If it works robustly, there is not much need to tamper with the threshold and number recent images to improve the detection. So if I were to improve the pipeline I would try to use a larger training set and a non-linear classifier. On my laptop the non-linear SVM was quite slow but the detection was much better. It might be possible to use fewer scales in combination with a non-linear SVM to get better results.

Another possibility is to use CNNs in order to introduce non-linearity and to improve the performance and robustness of the pipeline.
