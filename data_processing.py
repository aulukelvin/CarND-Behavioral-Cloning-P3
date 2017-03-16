import pandas as pd
import scipy.misc
import random
import numpy as np
import cv2
import time
import pickle 

#read data.txt
driving_log=pd.read_csv('./data/driving_log.csv').sample(frac=1.0)

driving_log.describe()

df_no_zero = driving_log[driving_log.steering != 0]
df_zero = driving_log[driving_log.steering == 0]

print("Non-zero steering data: %d"%(len(df_no_zero)))
print("Zero steering angle data: %d"%(len(df_zero)))
df = df_no_zero.append(df_zero.sample(frac=0.085))
print("Total data set: %d"%(len(df)))

df[['center','left','right','steering']].to_csv('./data/trimmed_driving_log.csv',index=False)

gallery = {}
def savetoGallery(string):
    assert(string is not None)
    string = string.strip()
    img = cv2.imread('./data/' + string)
    gallery[string] = img

images = driving_log[['center','left','right']].as_matrix()
for row in images:
    savetoGallery(row[0])
    savetoGallery(row[1])
    savetoGallery(row[2])

print('total number: {}'.format(len(gallery)))
f = open('./gallery.p', 'wb')   
pickle.dump(gallery, f)      
f.close() 