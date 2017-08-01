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


My pipeline consisted of the following steps:

1. Preprocessing


2. Model Architecture:

  Convolution gives us a small linear classifier for patch of the image with output of 28x28x6
  Pooling layer to decrease the size of the output and prevent overfitting

  We are using the Lenet architecture with the following Spec:
  layer 1:
  Convolution layer 1. The output shape should be 28x28x6.
  Activation 1. Your choice of activation function.
  Pooling layer 1. The output shape should be 14x14x6.
  layer 2:
  Convolution layer 2. The output shape should be 10x10x16.
  Activation 2. Your choice of activation function.
  Pooling layer 2. The output shape should be 5x5x16.
  layer 3:
  Flatten layer. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using tf.contrib.layers.flatten, which is already imported for you.
  Fully connected layer 1. This should have 120 outputs.
  Activation 3. Your choice of activation function.
  layer 4:
  Fully connected layer 2. This should have 84 outputs.
  Activation 4. Your choice of activation function.
  layer 5:
  Fully connected layer 3. This should have 10 outputs.


In the following image you can see an example of one of the test images, where the pipeline calulate the mean endpoints and slope for the left line (on blue) and for the right line (on green). then the function extend the lines (the black lines) where some of the endpoints exceeding the boundaries of the picture. The Region-Masking step (marked on yellow) will cut the outliers edges, and the final output will be as smoother as it can.
![]( https://github.com/shmulik-willinger/lane_line_detection/blob/master/readme_img/extend_lines.jpg?raw=true)


### Hyperparameter to tune

Stride - The amount by which the filter slides 
Filter depth -  For a depth of k, we need to connect each patch of pixels to k neurons in the next layer
CNN layers dimentions - Tradeoffs between model size and performance
learning rate -  
epochs -
batch size -
mean and standard deviation - changing the default parameters of tf.truncated_normal() can result in better performance

### Possible improvements to your pipeline

A possible improvement would be to create some values more dynamic. For example - in the reject outliers step, to calculate the outliners by some percentage of the total array instead of hardcoded values (which I worked hard for their fine-tuning)  

Another potential improvement could be to try recognize the lanes even if they are very blurred or on bad whether (sunny days and such)
