**Traffic Sign Recognition**

**Build a Traffic Sign classifier Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---
**Data Set Summary & Exploration**

The German Traffic Sign provided as the Dataset is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
'labels' is a 2D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.

I used the os library to load the sample images for the 'Train', 'Validation' and 'Test' data. With pandas library I calculated summary statistics of the traffic signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

Here is a sample of the dataset, displaying the first image from each class:

![]( https://github.com/shmulik-willinger/traffic_sign_classifier/blob/master/readme_img/dataset_samples.jpg?raw=true)

This are the exploratory visualization of the 'Train and 'Test' data set. It is a bar chart showing distribution of the imsage across the classes (different traffic signs):

![]( https://github.com/shmulik-willinger/traffic_sign_classifier/blob/master/readme_img/train_distribution.jpg?raw=true)

![]( https://github.com/shmulik-willinger/traffic_sign_classifier/blob/master/readme_img/test_distribution.jpg?raw=true)

**Pre-process the Data Set**

The distribution of signs between classes is very high, and the variance gets up from 210 samples for the lower class to 2250 samples to the highest one.  
I decided to generate additional data In order to raise the number of dataset samples. I Counted the lower & upper bounds of each class in order to  multiply each class images to raise the amount of training samples.

I used cv2 librarie to create Perspective Transform and rotation for the new augmented images.

All the images were also Transform to grayscale, since I noticed the accuracy of the model was higher this way (and it shorten the model runtime).
from shape (32,32,3) to (32,32,1)

I normalized the data to get higher validation and training accuracy (it accelerates the convergence).
from [0,255] to [0,1]

I divided the data into train (80%) and test (20%), and shuffled randomly each one of them.

After the preprocessing, my data set distribution was:

* Number of training examples = 71081
* Number of validation examples = 17769
* Number of testing examples = 12630
* Image data shape = (32, 32, 1)
* Number of classes = 43

Here is the a visualization bar chart showing the distribution of the imsage across the classes before and after the process:

![]( https://github.com/shmulik-willinger/traffic_sign_classifier/blob/master/readme_img/distribution_after_preprocessing.jpg?raw=true)

**Model Architecture**

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ?
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ...

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
