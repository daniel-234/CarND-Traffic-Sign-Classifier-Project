# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report
* Credits


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/traffic_sign_example.jpg "Example"
[image3]: ./examples/grayed_traffic_sign_example.jpg "Grayscaling"
[image4]: ./examples/rotated_traffic_sign_example.jpg "Rotation"
[image5]: ./examples/translated_traffic_sign_example.jpg "Translation"
[image6]: ./german_traffic_signs_resized/signs.jpg "Traffic Signs"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/daniel-234/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the Numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 * 32 * 3 = 3072
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed among the 43 classes in the training, validation and test set. 

![alt text][image1]

Here is a plot of a random signal from the training dataset. 

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because [experimental results](https://ieeexplore.ieee.org/document/7562656) in similar deep learning tasks showed that when color in images is not an essential feature, classification with grayscale images reaches higher accuracy in comparison with training with RGB images. This color downscaling also comes with the advantage of reduced computational costs. 

Then I normalized the image data because it is a good practice to scale the pixel values so that each pixel falls in a small range of values (from -1 to 1 here, but another common range is 0 to 1). 
This is because for most image data, the pixel values are integers between 0 and 255. Neural networks, though, process inputs using small weight values. Accordingly, inputs with large integer values can disrupt or slow down the learning process. 
It should also be noted that images with values in the new range can be viewed normally. 

Here is the image of the same traffic sign before and after normalization and grayscaling.

![alt text][image3]

As pointed out in [an article by Pierre Sermant and Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), the dataset provided by GTSRB presents a number of difficult challenges due to real-world variabilities such as viewpoint variations, physical damage, color fading and low input resolution. 
I tried some of the techniques they suggested to address some of these challenges and built a jittered dataset by adding transformed versions of the original one.

So first I generated a transformed version of the training dataset by rotating each image by a random angle between -15 and +15 degrees. This dataset was added to the original one, thus making a training dataset that was twice in size than before. 

Here is an example of the same traffic sign image  after a random rotation was applied.

![alt text][image4]

Then I generated another transformed version of the training dataset by shifting each image randomly by [-2, +2] pixels and added it to the original one. 
The resulting training dataset now yields 139196 samples.

Here is an example of the same traffic sign image  after a random translation was applied.

![alt text][image5]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten	    | 400      									|
| Fully connected		| 120        									|
| RELU					|	
| Fully connected		| 84        									|
| RELU					|	
| Fully connected		| 43        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 20 epochs and a batch size of 128. 
The Loss function I used is the Cross-entropy loss function and as optimizer I used the Adam optimizer. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.935 
* test set accuracy of 0.933

The architecture that was chosen was LeNet-5, the same that was implemented in the class labs. 

I believed it would be relevant to the traffic sign application because although it is more than 20 years old, it was very successful when presented for the recognition of handwritten characters. Even if the data for this project is quite different, I thought the analysis could benefit from a similar approach. This architecture is also very popular because of its simplicity. 
While working on this project, I tweaked the architecture trying to add Dropout layers, but noticed that the results didn't improve. Sometimes they even got worse. So my choice for the final implementation of the project was to use it as it is. 
 
The final results on accuracy after 20 epochs, as reported above , showed that the model trained well without overfitting on the training data. 
The results on validation (0.935) and test sets (0.933) were very satisfying. 

### Test a Model on New Images

#### 1. Choose at least five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![alt text][image6]

At first size the images I found shouldn't be difficult to classify, because they are in plain light and they occupy the majority of the space in the image. 
The first image is oriented sideways while the second upwards. Images 5, 6, 9, 10 have a background different from the blue sky. Images 3 and 4 appear to be quite clear, while images 7 and 8 seem a bit pixelated from the format conversion I applied (to have them all 32x32 in size). 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield					| Yield											|
| Turn right ahead      		| Turn right ahead   									| 
| No entry     			| No entry 										|
| Double curve					| Children crossing											|
| General caution					| General caution											|
| 70 km/h	      		| 70 km/h					 				|
| Road work			| Slippery Road      							|
| Roundabout mandatory			| Roundabout mandatory      							|
| Keep right			| Keep right      							|
| Children crossing			| General caution      							|


The model was able to correctly guess 7 of the 10 traffic signs, which gives an accuracy of 70%. This is slightly less than the accuracy on the test set of 93%. 

#### 3. Describe how certain the model is when predicting on each of the ten new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 31th cell of the Ipython notebook.

For the first, second, third, fifth, sixth and ninth images, the probability is 1.0, meaning the model is absolutely confident that the image contains a certain sign, that is indeed the right sign in the image. 
These signs, predicted with probability 1.0, are the following:

| Image number |    Prediction    |   Probability   |
| :----------: |    :--------:    |   :---------:   |
|      1       |      Yield       |     1.0     |
|      2       |   Turn right ahead    |     1.0     |
|      3       |   No entry    |     1.0     |
|      5       |   General caution    |     1.0     |
|      6       |   Speed limit (70km/h)    |     1.0     |
|      9       |   Keep right   |     1.0     |

For the fourth image, the model is confident that this is a Children Crossing sign (probability of 0.034), but the image contains instead a Double Curve sign. The top five softmax probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .934         			| Children crossing   									| 
| .065     				| Right-of-way at the next intersection 										|
| .0003					| General caution											|
| .0003	      			| Dangerous curve to the right					 				|
| .0003				    | Bumpy road      							|


For the seventh image, the model is confident that this is a Slippery Road sign (probability of 0.99), but the image contains instead a Road Work sign. The top five softmax probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Slippery road   									| 
| < .001     				| Wild animals crossing intersection 										|
| < .001				| Double curve											|
| < .001	      			| Dangerous curve to the right					 				|
| < .001				    | Road narrows on the right      							|

For the eighth image, the model is confident that this is a Roundabout Mandatory sign (probability of 0.99), and the image does indeed contain a Roundabout Mandatory sign. The top five softmax probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Roundabout mandatory   									| 
| .0005     				| Dangerous curve to the right intersection 										|
| .0004					| Right-of-way at the next intersection										|
| < .0001	      			| Children crossing					 				|
| < .0001				    | Speed limit (100km/h)      							|

For the tenth image, the model is confident that this is a General caution sign (probability of 0.99), but the image contains instead a Children crossing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| General caution   									| 
| 0.005     				| Children crossing 										|
| < .0001					| Bicycles crossing											|
| < .0001	      			| Priority road					 				|
| < .0001				    | No passing      							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I chose to visualize the feature maps for the 4th image, one of the three that were misclassified. 
It isn't really clear which characteristics made the model output the wrong prediction. It seems that looking at the feature maps from the first convolution, the sign edges were extracted together with the sign shape. 
The second part of the feature maps does not provide a clear interpretation. 

In regard to classification accuracy, I noticed that the most misclassified images were in the classes from 15 to 40. By looking at the distribution of classes in the first image of this writeup, you can see that these classes had approximately half the data of the first ones. By expanding the dataset, it made things even worse in terms of classes distribution. 
A solution could be to augment the portion of the dataset that is downsampled. 

I believe that the double curve sign is also heavily underrepresented in the dataset.  
Also it is not clear if German signs contemplate two different representations for the Double curve: 1) first curve on the left, second curve on the right; 2) first on the right followed by second curve on the left. 
The training dataset should be checked to verify it. In case there was such a problem, one solution could be to mirror all the images belonging only to class 21 (Double curve) to have a more robust dataset and train it properly on both types of double curves. This would also double the number of images for this class, which could help. 
Clearly most other signs can't be mirrored left to right. 



### Credits

[Machine Learning Mastery with Python](https://machinelearningmastery.com/machine-learning-with-python/)

[Using grayscale images for object recognition with convolutional-recursive neural networks](https://ieeexplore.ieee.org/document/7562656)

[Average grayscale from RGB image in Python](https://stackoverflow.com/questions/26201839/average-grayscale-from-rgb-image-in-python)

[Importing image data into Numpy arrays](https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays)

[Iterate through folder with Pillow to open images](#https://stackoverflow.com/questions/51178166/iterate-through-folder-with-pillow-image-open)

[How to create a 4D Tensor from images](https://stackoverflow.com/questions/50195967/how-can-i-create-a-4d-numpy-array-from-images)