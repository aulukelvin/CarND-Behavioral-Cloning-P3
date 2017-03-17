# SDC Behavioral Cloning Project
---
[//]: # (Image References)

[image1]: ./charts/rawdistribution.png "raw distribution"
[image2]: ./charts/augmented.png "augmented.png"
[image3]: ./charts/trimmeddistribution.png "trimmed distribution"
[image4]: ./charts/placeholder_small.png "Recovery Image"
[image5]: ./charts/placeholder_small.png "Recovery Image"
[image6]: ./charts/placeholder_small.png "Normal Image"
[image7]: ./charts/placeholder_small.png "Flipped Image"

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

Note: in order to recover the model.h5 please run:
Mac or Linux:
```sh
  cat model.h5a* > model.h5
```
Windows:
```sh
  copy model.h5aa + model.h5ab + model.h5ac > model.h5
```
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
```python
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
```
#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. I placed two dropout layers on top of the first and second fully-connected layers. The dropout servive rate is 0.7. 

#### 3. Model parameter tuning

I tried different batch sized and find out the model prefer larger dataset. So I decreased the images size from (320X160) to (64X64). The final batch_size is 450. I applied Adam optimizer to reduce the hassle tuning learning rate. I set the epoch to 20 and set early_stopping tolerance to 3. 

#### 4. Appropriate training data

In the data preprocessing, I found more than half of the data have the steering angle as zero. This imbalanced data may causes the model be overwhelmed by the zero steering angle samples and overlook other testing senarios. So I only kapt 0.085 of the zero steering angle data plus all other sample data. 

I noticed the model will be highly impacted by the imperfaction of the training data. Inspired by someone's 'live trainer' I developed a delta trainer which is re-train the almost-there model with hendpicked data set to make it able to pass through difficult scenarios.

Also in the drive.py, I intoduced a PD controller to reduce the jittering of the car drive.

The performance of the drive is heavily depends on the quality and quantity of the data. Because I'm a very terrible game player, I can't collect enough quality data by myself so I used the Udacity provided track 1 training data. I found out there're several issues with this dataset:
 * size of the training data. There're only 8036 sample images, which is quite small to produce good result.
 * the distribution. The data is significantly imbalanced. This will cause the less popular samples been ignored by the model. 
 * the quality. I found out the confusing zigzag driving style was very likely copped from wrong training data. Just for example, when some of the image shows car is actually turning, the steering angle in the training log is zero.
 * speed issue. In the previous data generator I read images directly from disk and do all normalization and augmentation on the fly. I found out the speed is extremely slow. It's not uncommon to have to spend more than 10 minutes to finish a single epoch. 
    
To handle the imbalanced data, I firstly built a trimmed data set which simply just drop out over 90% of the zero steering angle samples. After the trimming the distribution looks not that imbalabced. Before the data trimming, the raw steering angle was distributed like the following:
![raw distribution][image1]

After the trimming, the distribution is like below, we can see other nearly invisible in the previous histogram showing up in the new histogram:
![trimmed histogram][image3]

To make the training works smoothly on the small data set, I applied several augmentation to enlarge the training data set. In the training process, I reserved 15% of the data as validation data set, and use seperated data generators for training data and validation data. Several data augmentation techniques have been used in the training data generator: 
 * use left and right camera images;
 * randomly shift image horizontally;
 * randomly flip image over;
 * randomly augment brightness of the images
The augmented steering angle histogram is like the following:
![augmented histogram][image2]

To increase the data quality, I firstly use the trimmed dataset to train a rough model and then I find out the most significant difficult road sections and hand correct the steering label and save the new data set as the Delta data set. Then I fed the delta training data to fine train the model. I can see this can indeed enhance the performance of the trained model. I finally hand corrected 840 of the difficult section images and use them to fine train the model. Because this delta data set is pretty small, so I don't preserve dedicated validation set. I use 3 fold validation instead. This strategy works very well. It can help the model get better performance on the road.

Even after the Delta training, the car is still jittering significantly on the road. So I introduced a PD controller to help the drive, which also seems good.

To increase the speed, I load the image data in the memory so that the generator can save the time reading disk. Now the everage epoch run is only 25 seconds, a 20X enhancement.

### Training Process
Besides the train.py, there're three notebooks: 
 * BehaviourCloning-EDA and preprocessing.ipynb, The notebook for explore the distribution of the data, generate trimmed training data(keep only 8.5% of the raw data), and also generate pickled image data source. The reason I use image data source is I noticed loading image directly from disk in the data generator is extremely inefficient. Even with my SSD, pre-loading image into memory can still yield 20 times faster training.
 * BehaviourCloning.ipynb, the main part of the model traing, which load the trimmed driving log csv file and the pickled image data, then do split train/valid data set.It also contains aumentation logic, model definition, and a plot do evaluate the performance of the model and the distribution of the steering angles produced by the generator.
 * BehaviourCloning-DeltaTrainer.ipynb. The logic of the Delta trainer is the same as the main model trainer. The only difference is it's been used when there're additional training data to get the model polished for some difficult scenarios. In my case I hand crafted 840 samples from the original testing data and fed them into the Delta trainer. The Delta trainer can really helps the car to perform better.
    
### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I firstly used a pretrained VGG16 model to run through the track 1 sample data. Surprisingly I found out the car can already drive through the whole lap. But I also realized there're several problems which took me the rest of the project time to fix: 
 * The training result is not repeatable. Sometimes a trained model can work well sometimes not so much, although the data set and the parameters are the same. I believe the problem is the data set is too small. Every time I train the model the train/test data set will be randomly splited, sometimes some very important data will be put into validation data set instead of the training set so that the model will unable to learn enough information. 
 * The car jittering a lot on the road. Two factors may lead to this issue. Firstly, the training data has lot of noise, for example steering angle was not align with the image; or even worse, the training data set has recorded lot of jittering drive. So that the model will copy the behaviour of the bad driving data. And the last factor, just like in reality, the steering angle should also depends on the car speed. The higher the speed, the stabler the steering must be. But the current model doesn't take speed into training so the model can only work under certain limit.   

So I augmented the training data to enlarge the data set and it works really well. I then fine trained the model use the delta trainer to enhance the driving performance. I adjusted the PI controller in the drive.py to make the car smoothly speed up untill reached the speed limit. Finally, inspired by the PI controller, I developed a PD controller to reduce the car jittering. 

#### 2. Final Model Architecture

Here is a description of the architecture:

|Layer (type)                   |  Output Shape         |   Param #   |  Connected to         |             
|-------------------------------|-----------------------|-------------|-----------------------|
|input_1 (InputLayer)           |  (None, 64, 64, 3)    |   0         |                       |                  
|block1_conv1 (Convolution2D)   |  (None, 64, 64, 64)   |   1792      |    input_1[0][0]      |              
|block1_conv2 (Convolution2D)   |  (None, 64, 64, 64)   |   36928     |    block1_conv1[0][0] |              
|block1_pool (MaxPooling2D)     |  (None, 32, 32, 64)   |   0         |    block1_conv2[0][0] |              
|block2_conv1 (Convolution2D)   |  (None, 32, 32, 128)  |   73856     |    block1_pool[0][0]  |              
|block2_conv2 (Convolution2D)   |  (None, 32, 32, 128)  |   147584    |    block2_conv1[0][0] |              
|block2_pool (MaxPooling2D)     |  (None, 16, 16, 128)  |   0         |    block2_conv2[0][0] |              
|block3_conv1 (Convolution2D)   |  (None, 16, 16, 256)  |   295168    |    block2_pool[0][0]  |              
|block3_conv2 (Convolution2D)   |  (None, 16, 16, 256)  |   590080    |    block3_conv1[0][0] |              
|block3_conv3 (Convolution2D)   |  (None, 16, 16, 256)  |   590080    |    block3_conv2[0][0] |              
|block3_pool (MaxPooling2D)     |  (None, 8, 8, 256)    |   0         |    block3_conv3[0][0] |              
|block4_conv1 (Convolution2D)   |  (None, 8, 8, 512)    |   1180160   |    block3_pool[0][0]  |              
|block4_conv2 (Convolution2D)   |  (None, 8, 8, 512)    |   2359808   |    block4_conv1[0][0] |              
|block4_conv3 (Convolution2D)   |  (None, 8, 8, 512)    |   2359808   |   block4_conv2[0][0]  |             
|block4_pool (MaxPooling2D)     |  (None, 4, 4, 512)    |   0         |    block4_conv3[0][0] |              
|block5_conv1 (Convolution2D)   |  (None, 4, 4, 512)    |   2359808   |    block4_pool[0][0]  |              
|block5_conv2 (Convolution2D)   |  (None, 4, 4, 512)    |   2359808   |    block5_conv1[0][0] |              
|block5_conv3 (Convolution2D)   |  (None, 4, 4, 512)    |   2359808   |    block5_conv2[0][0] |              
|block5_pool (MaxPooling2D)     |  (None, 2, 2, 512)    |   0         |    block5_conv3[0][0] |              
|
|flatten_1 (Flatten)            |  (None, 2048)         |   0         |    block5_pool[0][0]  |              
|dense_1 (Dense)                |  (None, 512)          |   1049088   |    flatten_1[0][0]    |              
|dropout_1 (Dropout)            |  (None, 512)          |   0         |    dense_1[0][0]      |              
|dense_2 (Dense)                |  (None, 128)          |   65664     |    dropout_1[0][0]    |              
|dropout_2 (Dropout)            |  (None, 128)          |   0         |    dense_2[0][0]      |              
|prediction (Dense)             |  (None, 1)            |   129       |    dropout_2[0][0]    |              

Total params: 15,829,569
Trainable params: 1,114,881
Non-trainable params: 14,714,688
---
## What haven't tried
 * The track2. Limited by my time and my PC gaming skill, it's really hard for me to collect good quality training data. So I have to regretfully give up the track 2.
 * Maybe I can put car speed into the training process to make the car run smoother at high speed.
 * If I transfer the image into HSV color space and only train the model use the S channel, maybe the computation can becomes easier and the model may more tolerant to lighting change. 
