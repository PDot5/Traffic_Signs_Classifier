# Self Driving Car Traffic Sign Classifier

---
## Goals:

The goals of this project are as follows:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---
# Step 1: Data Exploration
### 1. Summary of Dataset:

In this project, numpy was used for shape in order to calculate the dataset and display the results through output. Import pickled data also contained a resized verision (32 X 32) of the dataset images.

### 2. Exploratory Visualization of Dataset:
* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

### 3. Normalized Distribution of Traffic Sign Classes:
Here are some links to visual data:
- Normalized Distribution of Traffic Sign Classes
![alt text](https://github.com/PDot5/CarND-Traffic-Sign-Classifier-Project/Markdown_Image_Ref/NormalizedzDistributionClasses.png "Normalized Distribution of Traffic Sign Classes")

- Sample of Training Set
![alt text](https://github.com/PDot5/CarND-Traffic-Sign-Classifier-Project/Markdown_Image_Ref/SampleTrainingData.png "Sample Training Data")

- 43 Classes Visualization: (Using signname.csv and append the labels to the images)
![alt text](https://github.com/PDot5/CarND-Traffic-Sign-Classifier-Project/Markdown_Image_Ref/43ClassesVisualization.png "Visualization of 43 Traffic Sign Classes")

# Step 2: Design and Test a Model Architecture
### 1. Designing and Implementing:
In designing and implementing a deep learning model that learns to recognize traffic signs, the training and test data was using to test the model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset
Using the LeNet solution from the lecture, the validation accuracy was measured.
![alt text](https://github.com/PDot5/CarND-Traffic-Sign-Classifier-Project/Markdown_Image_Ref/ValidationAccuracy.png "Validation Accuracy")
The accuracy rate was greater than 98% and the model was saved.

### 2. Preprocessed Data:
First I reshaped the images giving an updated shape of (32x32x3). I explored specific testing data and output the classes from the specific images I downloaded from the internet in order to visualize the training data.
![alt text](https://github.com/PDot5/CarND-Traffic-Sign-Classifier-Project/Markdown_Image_Ref/SpecificClassesOutput.png "Specific Classes Output")

Additionally, I used a couple techniques for grayscaling and normalizing the data. (One technique for each would have sufficed but I was experimenting). Next I created a preprossed function to call on grayscaling and normalization.

Before running the dataset through the convolutional newtwork, I shuffled the training data. This ensures that there is no bias during the training while also preventing the model from learning the order of the training. In addition, it also helps with the speed of training convergence.

### 3. Model Architecture
There were a few hyperparameters to set, such as EPOCHS, BATCH_SIZE, learning rate, and dropout rate. Randomly defining variables for weights and biases for each layer were also determined such as mu and sigma.
The Model itself is represented in steps below using TensorFlow:

| Layers                    | Description       |
| --------------------------|:-------------------:|
| Input                     | (32X32X1) RGB image |
| Convolution 1: 5X5 filter | Input depth 3, output depth 6 (5,5,,3,6) |
| Solution Activation       | ReLu  |
| Solution Pooling          | Max Pooling Input 28X28X14 Output 14X14X6 |
| Convolution 2:            | Input depth 6, output depth 16 (5,5,6,16) |
| Solution Activation       | ReLu  |
| Solution Pooling          | Max Pooling Input 10X10X16 Output 5X5X16 |
| Solution Flatten          | Flatten Convolution 2 Input 5X5X16 Output 400 |
| Solution Layer 3          | Fully Connected 1. Input 400 Output 120 |
| Solution Activation       | ReLu on Fully Connected 1 |
| Solution Layer 4          | Fully Connected 2. Input 120 Output 84 |
| Solution Activation       | ReLu on Fully Connected 2 |
| Add Dropout               | keep_prob |
| Solution Layer 5          | Fully Connected Matrix Multipliaction Input 84 Output 43 |

### 4. Train, Validate, and Test Model
Next, create a placeholders for a batch of input images (x), and a placeholder for output images (y). I chose to set a learning rate of 0.0021 and then during the evalutation process, one-hot encoding will be applied to the images.

Once the code is ran, the training will begin and calculate the validation accuracy and save the data in my specified file location. I chose to using a dropout rate (keep_prob of 0.92) for overfitting. The final results:

* Validation Accuracy: 98%
* Test Accuracy: 92%

# Step 3: Test Model on New Images
After downloading 5 images from the internet, I created a function to read in the images and applied the grayscale technique. I output a visual representation of both the color images and the grayscale.

In addition to this, I resized the images to (32x32x3) and normalized them. With the new test set I recalled the model.meta data and evaluated the new images with the testing data. My current results are below:
![alt text](https://github.com/PDot5/CarND-Traffic-Sign-Classifier-Project/Markdown_Image_Ref/DownloadedImagesColor_grayscale.png "Downloaded Images Color & Grayscale")

In addition, I wanted to display the results from the processed test set:
![alt text](https://github.com/PDot5/CarND-Traffic-Sign-Classifier-Project/Markdown_Image_Ref/ConvolutionalNetworkTestSet.png "Processed Convolutional Network Test Set")

This model has calculated a 60% Test Set Accuracy, only getting 3 out of the 5 images correct:

![alt text](https://github.com/PDot5/CarND-Traffic-Sign-Classifier-Project/Markdown_Image_Ref/Test_Set_Accuracy_Prediction.png "Test Set Accuracy Prediction")

## Potential Shortcomings
Perhaps the low percentage in the Test Set Accuracy of only 60% could be attributed to the high dropout rate could have contributed to the results for overfitting. Additionally I could have determined a better learning rate.

## Suggest possible improvements to your pipeline
* I believe that introducing some more preprocessed techniques may help resolve some of the issues
* Determine best Learning Rate
* Determing best Dropout Rate
