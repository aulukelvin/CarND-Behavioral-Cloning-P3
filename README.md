**SDC Behavioral Cloning Project**

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network (due to GitHub file size limit the model.h5 was splitted into 3 parts which will have to be combined together as explained below in detail)
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

In the project I tried VGG16 pretrained model, NVIDIA model, Comma AI model, and a modified Comma AI model. I found out the NVIDIA and Comma AI models have significant lower complexity but on my computer, the speed are almost the same while the VGG16 can produce slightly better running performance. So the final model is VGG16 pretained model with top fully connected layers by three trainable fully connected layers: width of the fist one is 512 and the middle layer is 128. The top layer is a linear output layer to produce the prediction. 
The model includes ELU layers to introduce nonlinearity. The input size is (64x64) and the data is normalized to between -0.5 ~ 0.5.
The code is as the following:
'''python
def getVGG16model(input_shape=(64,64,3)):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='elu')(x)
    x = Dropout(0.6)(x)
    x = Dense(128, activation='elu')(x)
    x = Dropout(0.7)(x)
    prediction = Dense(1, name='prediction')(x)

    model = Model(input=base_model.input, output=prediction)

    optimizer = Adam()
    model.compile(loss='mse', optimizer=optimizer)
    
    return model

#### 2. Attempts to reduce overfitting in the model

In the data preprocessing, I found more than half of the data have the steering angle as zero. This imbalanced data may causes the model be overwhelmed by the zero steering angle samples and overlook other testing senarios. So I only kapt 0.085 of the zero steering angle data plus all other sample data. 

The model contains dropout layers in order to reduce overfitting. I placed two dropout layers on top of the first and second fully-connected layers. The dropout servive rate is 0.7. 

In the training process, I reserved 15% of the data as validation data set, and use seperated data generators for training data and validation data. Several data augmentation techniques have been used in the training data generator: 
    - use left and right camera images;
    - randomly shift image horizontally;
    - randomly flip image over;
    - randomly augment brightness of the images

I noticed the model will be highly impacted by the imperfaction of the training data. Inspired by someone's 'live trainer' I developed a delta trainer which is re-train the almost-there model with hendpicked data set to make it able to pass through difficult scenarios.

Also in the drive.py, I intoduced a PD controller to reduce the jittering of the car drive.

#### 3. Model parameter tuning

I tried different batch sized and find out the model prefer larger dataset. So I decreased the images size from (320X160) to (64X64). The final batch_size is 450. The epoch was set to 20. The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

The performance of the drive is heavily depends on the quality and quantity of the data. Because I'm a very terrible game player, I can't collect enough quality data by myself so I used the Udacity provided track 1 training data. I found out there're several issues with this dataset:
    - the distribution. The data is significantly imbalanced. This will cause the less popular samples been ignored by the model. So I only kept 0.085 of the data set.
    - the quality. I found out the confusing zigzag driving style was very likely copped from wrong training data. Just for example, when the image shows car is actually turning, sometimes the steering angle is zero, or sometimes the car started to jittering from one side to the other. So I find out the significant problematic sections and hand correct the steering label then fed the delta training data to re-train the model. I can see this can indeed enhance the performance of the trained model. 

### Training Process
Besides the train.py, there're three notebooks: 
    - BehaviourCloning-EDA and preprocessing.ipynb, The notebook for explore the distribution of the data, generate trimmed training data(keep only 8.5% of the raw data), and also generate pickled image data source. The reason I use image data source is I noticed loading image directly from disk in the data generator is extremely inefficient. Even with my SSD, pre-loading image into memory can still yield 10 times faster training.
    - BehaviourCloning.ipynb, the main part of the model traing, which load the trimmed driving log csv file and the pickled image data, then do split train/valid data set.It also contains aumentation logic, model definition, and a plot do evaluate the performance of the model and the distribution of the steering angles produced by the generator.
    - BehaviourCloning-DeltaTrainer.ipynb. The logic of the Delta trainer is the same as the main model trainer. The only difference is it's been used when there're additional training data to get the model polished for some difficult scenarios. In my case I hand crafted 840 samples from the original testing data and fed them into the Delta trainer. The Delta trainer can really helps the car to perform better.
    

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
