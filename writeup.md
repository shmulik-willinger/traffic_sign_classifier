# **Finding Lane Lines on the Road**

## The goals


 The goals of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Reflection of the work and results in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Pipeline description


My pipeline consisted of 8 steps:
1. First, I converted the images to grayscale

2. Using Gaussian smoothing which is essentially a way of suppressing noise and spurious gradients by averaging

3. Using Canny edge detection. The algorithm will first detect strong edge (strong gradient) pixels above the high_threshold, and reject pixels below the low_threshold. Next, pixels with values between the low_threshold and high_threshold will be included as long as they are connected to strong edges. The output edges is a binary image with white pixels tracing out the detected edges and black everywhere else

4. Region of interest/ Region-Masking, to represent the vertices of a quadrilateral region that I would like to retain for my color selection, while masking everything else out.

5. Hough Transform - using OpenCV function HoughLinesP which gets the output from Canny and the result will be array of lines, containing the endpoints (x1, y1, x2, y2) of all line segments detected by the transform operation

6. Drawing the lines - In order to draw a single line on the left and right lanes, I created a function that get the hough array and perform the following:
  * separate the hough array into two arrays, for the left and right lines, according to the slope (derived by the endpoints by using ((y2-y1)/(x2-x1)))
  * reject outliers slopes for each line according to a min and max values
  * calculate the mean slope for each line
  * extend the lines to fill all the slope, in this case I used   simple multiplication with big negative and positive values (\*400 and \*(-400))
  * calculate the mean of the new lines result with the previews lines for better smoothing
  * raws lines with color and combine it with the original image

7. Create a loop that will take all the tests images and run the pipeline processing on them. The output will be sent to the output folder

8.  run the processing on the videos. The output can be found on the test_videos_output folder

In the following image you can see an example of one of the test images, where the pipeline calulate the mean endpoints and slope for the left line (on blue) and for the right line (on green). then the function extend the lines (the black lines) where some of the endpoints exceeding the boundaries of the picture. The Region-Masking step (marked on yellow) will cut the outliers edges, and the final output will be as smoother as it can.
![]( https://github.com/shmulik-willinger/lane_line_detection/blob/master/readme_img/extend_lines.jpg?raw=true)


### Potential shortcomings with the current pipeline


One potential shortcoming would be what would happen when the lanes on the road are blurred or erased, sometimes they are even not exist, and the algorithm will have difficulty to draw them

Another shortcoming could be the sharp turns we have in roads. In this cases the pipeline may not work well.


### Possible improvements to your pipeline

A possible improvement would be to create some values more dynamic. For example - in the reject outliers step, to calculate the outliners by some percentage of the total array instead of hardcoded values (which I worked hard for their fine-tuning)  

Another potential improvement could be to try recognize the lanes even if they are very blurred or on bad whether (sunny days and such)
