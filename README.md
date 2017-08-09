# **Build a Traffic Sign Recognition Program**
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

Neural networks is a beautiful biologically-inspired programming paradigm which enables a computer to learn from observational data.
Deep learning is a powerful set of techniques for learning in neural networks and deep learning, currently provide the best solutions to many problems, including image recognition.

A convolutional neural networks (CNN) technique consist of several layers with different filters, where each one pick up different qualities of a patch. The subsequent layers tend to be higher levels in the hierarchy and generally classify more complex ideas, while eventually the CNN classifies the image by combining the larger, more complex objects, grouping together adjacent pixels and treating them as a collective
The CNN learns all of this on its own and also helps us with translation invariance and gives us smaller, more scalable model.

This project is using deep neural networks and convolutional neural networks to classify traffic signs. We will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, we will then try out the model on images of German traffic signs that we find on the web.

More details can be found in the [writeup_report.md](https://github.com/shmulik-willinger/traffic_sign_classifier/blob/master/writeup.md) file.

### Dependencies

This project requires **Python 3.5** and the following Python libraries installed:

- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [OpenCV](https://pypi.python.org/pypi/opencv-python#)
- [Sklearn](scikit-learn.org/)
- [Pandas](pandas.pydata.org/)
- [TensorFlow](http://tensorflow.org)


### Dataset

1. Download the dataset. You can download the pickled dataset in which we've already resized the images to 32x32 [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d53ce_traffic-sign-data/traffic-sign-data.zip).

2. Clone the project and start the notebook.
```
git clone https://github.com/shmulik-willinger/traffic_sign_classifier
cd traffic-signs
jupyter notebook traffic_Signs_Recognition.ipynb
```
3. Follow the instructions in the `traffic_Signs_Recognition.ipynb` notebook.
