# **Traffic Sign Recognition**

**Build a Traffic Sign classifier Project**

## The goals

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---
## Data Set Summary & Exploration

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

This are the exploratory visualization of the 'Train and 'Test' data set. The bar chart showing distribution of the images across the classes (different traffic signs):

![]( https://github.com/shmulik-willinger/traffic_sign_classifier/blob/master/readme_img/train_distribution.jpg?raw=true)

![]( https://github.com/shmulik-willinger/traffic_sign_classifier/blob/master/readme_img/test_distribution.jpg?raw=true)

## Pre-process the Data Set

The distribution of signs between classes is very high, and the variance gets up from 210 samples for the lower class to 2250 samples for the highest one.  
I decided to generate additional data In order to raise the number of dataset samples. I Counted the lower & upper bounds of each class in order to multiply each class images with respective to the number of original count, to raise the amount of training samples.

I used cv2 library to create Perspective Transform and rotation for the new augmented images.

All the images were also Transformed to grayscale, since I noticed the accuracy of the model was higher this way, and of couse it also shorten the model runtime.
from shape (32,32,3) to (32,32,1)

Here is a sample of the grayscale dataset, displaying the first image from each class:

![]( https://github.com/shmulik-willinger/traffic_sign_classifier/blob/master/readme_img/dataset_samples_gray.jpg?raw=true)

I normalized the data to get higher validation and training accuracy (it accelerates the convergence).
from [0,255] to [0,1]

I divided the data into train (80%) and test (20%), and shuffled randomly each one of them.

After the preprocessing, my data set distribution was:

* Number of training examples = 71081
* Number of validation examples = 17769
* Number of testing examples = 12630
* Image data shape = (32, 32, 1)
* Number of classes = 43

Here is the visualization bar chart showing the distribution of the images across the classes before and after the process:

![]( https://github.com/shmulik-willinger/traffic_sign_classifier/blob/master/readme_img/distribution_after_preprocessing.jpg?raw=true)

## Model Architecture

My model consisted of the following layers:
<!---
| Layer | Component        	|     Input	      	| Output |
|:---------------------:|:---------------------------------------------:|
| Convolution layer 1 | 2D Convolution layer with 'VALID' padding, filter of (5x5x1) and Stride=1 | (32,32,1) 	| (28,28,6)|
|   	| ReLU Activation 	| (28,28,6) | (28,28,6)|
| Convolution layer 2|	2D Convolution layer with 'SAME' padding, filter of (3x3x6) and Stride=1	|(28,28,6) | (28,28,6)|
|    	| Max pooling	with 'VALID' padding, Stride=2 and ksize=2	| (28,28,6) | (14,14,6)|
| Convolution layer 3   | 2D Convolution layer with 'VALID' padding, filter of (5x5x12) and Stride=1	| (14,14,6)| (10,10,16)|
| 	|  ReLU Activation  		|(10,10,16)|(10,10,16)|
| 	| Max pooling	with 'VALID' padding, Stride=2 and ksize=2	|(10,10,16)|(5,5,16)|
| Fully connected	layer 1	| Reshape and Dropout|(5,5,16)| 400|
| | Linear combination WX+b |400| 120|
| | ReLU and Dropout |120| 120|
| Fully connected	layer 2	| Linear combination WX+b|120| 84|
| | ReLU and Dropout |84| 84|
| Fully connected	Output layer	| Linear combination WX+b|84| 43 |
-->
![]( https://github.com/shmulik-willinger/traffic_sign_classifier/blob/master/readme_img/model.jpg?raw=true)

**Model training**

To train the model, I used LeNet architecture as a baseline, with additional 2D Convolution layer and a couple of parameter tunning.
batch size was set to 100, and I also used 100 epochs.
The learning rate (0.001), mean (0) and sigma (0.1) were left with their default values.   

**Approach taken for finding the solution**

I started with LeNet architecture and the train accuracy wasn't high enough, so I start with changing the default parameters of the batch and epochs which improved the results.
I applied grayscale on the dataset which give some better results and also the model is much more faster.
After adding more images to the dataset (in the preprocessing step) I also observed better results.
Adding the dropout function after each layer also improving the accuracy ('Max pooling' steps dosen't need it since it already performing dropout)
ReLU activation function produced better results then sigmoid.
I found that splitting the LeNet first Convolution layer to two 2D Convolution layers helps getting higher accuracy.
I left the learning rate and the Mean as is since after trying to change them a little bit I didn't get better results.
There are lots options to change parameters in the Model - padding, stride, filters, connected shapes, weight and bias initialization and more. I tunned them many times till I got results to my satisfaction.
AdamOptimizer was set instead of the GradientDescentOptimizer since its using 'momentum'.

Training was performed on an Amazon g2.2xlarge GPU server, and it took about 16 minutes.

My final model results were:
* training set accuracy of 96.2%
* validation set accuracy of 95.2%
* test set accuracy of 93.4%

![]( https://github.com/shmulik-willinger/traffic_sign_classifier/blob/master/readme_img/validation_accuracy.jpg?raw=true)

The validation accuracy become balanced at around the 60 epoch, so in order to avoid overfitting and save training time I reduced the number of epochs from 100 to 60.

A low accuracy on the training and validation sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.


## Test the Model on New Images


Here are 10 German traffic signs that I found on the web:

![]( https://github.com/shmulik-willinger/traffic_sign_classifier/blob/master/readme_img/new_images.jpg?raw=true)

Some of the image (image:0, and image:5) might be difficult to classify because they have two signs combined in each of them. The classifies might label it as one of the signs or non of them.
The third image (image:2) has a sign that is not part of the train dataset classes, meaning the classifier is lack of information about it and probably will not label it right.
The other images has the stick of the sign occupies a large part of the picture, and since the images needs to get through the same preprocessing stage (grayscale, normalization, resizing to 32x32 etc.) before running the model on them, the sign size in the images might be significantly smaller then the train dataset.

Also, the images background and signs brightness along with their rotation angles can be challenging.


**Model's predictions on the New Images**

Here are the results of the prediction:

<!---
| Image	number	|     Sign name| Prediction	  	|
|:---------------------:|:---------------------------------------------:|
| Image 0 | Stop Sign    | Stop sign  	|
| Image 1  | Keep right  | Keep right 		|
| Image 2  | Caution Falling cows		| Ahead	only |
| Image 3  | Double curve		| Dangerous curve to the right	|
| Image 4  | Bumpy road	| Road work	|
| Image 5  | Speed limit 30	| Dangerous curve to the left		|
| Image 6  | Road work	| Road work	|
| Image 7  | Speed limit 70	| Speed limit 70	|
| Image 8  | Children crossing	| Speed limit 50	|
| Image 9  | Speed limit 30	| Speed limit 30	|
-->

![]( https://github.com/shmulik-willinger/traffic_sign_classifier/blob/master/readme_img/predictions_table.jpg?raw=true)

![]( https://github.com/shmulik-willinger/traffic_sign_classifier/blob/master/readme_img/prediction_performance.jpg?raw=true)

The model was able to correctly predict 5 of the 10 traffic signs, which gives an accuracy of 50%. This compares favorably to the accuracy on the test set of 93.4%. The model didn't perform well on half of the new images.

The images that were not included at all in the training dataset (no suitable class) were labled incorrectly. The other images were processed differently (angles, cropped etc.) so the model also failed to classify some of them correctly.
I noticed that the model classify 20% og the images as 'Dangerous curve', while one of this images is understandable (image:3 , Double curve) while it's quite similar, while I was disappointed from image:5 (Speed limit 30) that the model failed on.

![]( https://github.com/shmulik-willinger/traffic_sign_classifier/blob/master/readme_img/new_images_predictions.jpg?raw=true)

**softmax probabilities prediction**

The code for making predictions on my final model is located in the 63th cell of the Ipython notebook.

We are looking at the softmax probabilities for each prediction to display how certain the model is when predicting each of the images.
Below we can see the visualizations results of the top 5 softmax probabilities for each image along with the sign type of each probability:

![]( https://github.com/shmulik-willinger/traffic_sign_classifier/blob/master/readme_img/softmax.jpg?raw=true)

For almost all the images the predictor was very certain with probability of more than 50%, even when the results were wrong.

I noticed that for some of the images - I got different predictions each time I ran the model (also with high probability on them) which is quite strange

For the images that were predict correctly we can observe probability of more than 90% which is preety satisfying.

Augmenting the training set definitely help improve model performance. I used rotation and translation as data augmentation techniques, after searching the web for new images I noticed that it's important to use also zoom, flips and color perturbation

### (Optional) Visualizing the Neural Network
I added the code that suppose to run the FeatureMap function, but I ran into some issues with calling the tensorflow session inside the function (guess I'm missing something there).
Hope to come back to it later on
