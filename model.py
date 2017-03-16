from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, MaxPooling2D
from keras.regularizers import l2, activity_l2
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation, Reshape
import numpy as np
import pandas as pd
import scipy.misc
import random
import numpy as np
import time
import cv2
import pickle
from sklearn.utils import shuffle

#loading trimmed driving log
driving_log=pd.read_csv('./data/trimmed_driving_log.csv').sample(frac=1.0)

driving_log.describe()

#loading images
f = open('./gallery.p', 'rb')   
gallery = pickle.load(f)      
f.close() 
print(len(gallery))

#brightness augmentation
def augment_brightness(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    
    image[:,:,2] = image[:,:,2]*random_bright
    return image
#image shifting augmentation
def trans_image(image,steer,trans_range=50, trans_y=False):
    """
    translate image and compensate for the translation on the steering angle
    """
    assert(image is not None)
    rows, cols, chan = image.shape
    
    # horizontal translation with 0.008 steering compensation per pixel
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*.4
    
    # option to disable vertical translation (vertical translation not necessary)
    if trans_y:
        tr_y = 40*np.random.uniform()-40/2
    else:
        tr_y = 0
    
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    
    return image_tr,steer_ang

pts1 = np.float32([[140,60],[180,60],[0,100],[320,100]])
pts2 = np.float32([[140,0],[180,0],[0,120],[320,120]])

M = cv2.getPerspectiveTransform(pts1,pts2)
#resize image to 64 x 64, trim off sky area
def transform(img):
    dst = cv2.warpPerspective(img,M,(320,160))
    dst = cv2.resize(dst, (64,64))
    return dst / 255 - 0.5
#train data generator
y_train_log = []
def generate_batch_samples(df, batch_size=128):    
    camera_shift_rate = {0:0, 1:0.27, 2:-0.27}
    while 1:
        df = df.sample(batch_size)
        source_arr = df.as_matrix()
        
        batch_x, batch_y = [], []
        for row in source_arr:
            # randomly pick between left right and center camaras
            rnd = random.randint(0,2)
            
            image = gallery[row[rnd].strip()]
            
            # compensate -0.2 for right camera and 0.2 for left camera 
            angle = row[3] + camera_shift_rate[rnd] 
            angle = max(-1, angle)
            angle = min(1, angle)
            
            # horizontally random shift the image
            image, angle = trans_image(image, angle)
            
            # brightness augmentation
            if random.random() >0.5:
                image = augment_brightness(image)
                
            #randomly flip
            if random.random() >0.5:
                image = cv2.flip(image, 0)
                angle = -angle
            
            batch_x.append(transform(image))
            batch_y.append(angle)
            y_train_log.append(angle)
        yield shuffle(np.array(batch_x), np.array(batch_y))
                
#validation data generator    
y_valid_log = []
def generate_batch_valid(df, batch_size=128):
    while 1:
        df = df[['center','steering']].sample(batch_size)
        source_arr = df.as_matrix()
        
        batch_x, batch_y = [], []
        for row in source_arr:
            image = gallery[row[0].strip()]
            angle = row[1]
            
            batch_x.append(transform(image))
            batch_y.append(angle)
            y_valid_log.append(angle)
        yield shuffle(np.array(batch_x), np.array(batch_y))
            

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

def get_commaai_model(input_shape=(64,64,3)):
    model = Sequential()
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same",input_shape=input_shape))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

def nvidia_model(input_shape=(64,64,3)):
    INIT='glorot_uniform' # 'he_normal', glorot_uniform
    keep_prob = 0.2
    reg_val = 0.01
    
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init=INIT,input_shape=input_shape, W_regularizer=l2(reg_val)))
    # W_regularizer=l2(reg_val)
    model.add(ELU())
    model.add(Dropout(keep_prob))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init=INIT))
    model.add(ELU())
    model.add(Dropout(keep_prob))
    
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init=INIT))
    model.add(ELU())
    model.add(Dropout(keep_prob))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init=INIT))
    model.add(ELU())
    model.add(Dropout(keep_prob))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init=INIT))
    model.add(ELU())
    model.add(Dropout(keep_prob))
    
    model.add(Flatten())

    model.add(Dense(100))
    model.add(ELU())
    model.add(Dropout(0.2))
    
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dropout(0.2))
    
    model.add(Dense(10))
    model.add(ELU())
    
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse") # , metrics=['accuracy']
    
    return model


from keras.callbacks import EarlyStopping
epochs = 20
batch_size = 450

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

from sklearn.model_selection import train_test_split
df_train, df_valid = train_test_split(driving_log, train_size=0.85)

train_gen = generate_batch_samples(df_train, batch_size=batch_size)
valid_gen = generate_batch_valid(df_valid, batch_size=batch_size)

model = getVGG16model(input_shape=(64,64,3))
#model = nvidia_model()
#model = get_commaai_model()
print(model.summary())
# train model with generator
model.fit_generator(
    train_gen,
    samples_per_epoch=batch_size*30, nb_epoch=epochs,
    validation_data=valid_gen,
    nb_val_samples=batch_size,
    callbacks=[early_stopping]
)

model.save('model.h5')
