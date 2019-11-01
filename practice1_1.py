# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 21:14:29 2019

@author: shawn
"""

import numpy as np # We'll be storing our data as numpy arrays
import os # For handling directories
from PIL import Image # For handling the images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # Plotting

def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    print(f'pred_array: {pred_array}')
    result = gesture_names[np.argmax(pred_array)]
    print(f'Result: {result}')
    print(max(pred_array[0]))
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(result)
    return result, score

def prepareData():
    lookup = dict()
    reverselookup = dict()
    count = 0
    for j in os.listdir('leapGestRecog/00/'):
        if not j.startswith('.'): # If running this code locally, this is to 
                                  # ensure you aren't reading in hidden folders
            lookup[j] = count
            reverselookup[count] = j
            count = count + 1
            
    x_data = []
    y_data = []
    datacount = 0 # We'll use this to tally how many images are in our dataset
    for i in range(0, 10): # Loop over the ten top-level folders
        for j in os.listdir('leapGestRecog/0' + str(i) + '/'):
            if not j.startswith('.'): # Again avoid hidden folders
                count = 0 # To tally images of a given gesture
                for k in os.listdir('leapGestRecog/0' + 
                                    str(i) + '/' + j + '/'):
                                    # Loop over the images
                    img = Image.open('leapGestRecog/0' + 
                                     str(i) + '/' + j + '/' + k).convert('L')
                                    # Read in and convert to greyscale
                    img = img.resize((320, 120))
                    img=np.array(img)
                    ret, binary = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
                    arr = np.array(binary)
#                    arr = np.array(img)
                    x_data.append(arr) 
                    count = count + 1
                y_values = np.full((count, 1), lookup[j]) 
                y_data.append(y_values)
                datacount = datacount + count
    x_data = np.array(x_data, dtype = 'float32')
    y_data = np.array(y_data)
    y_data = y_data.reshape(datacount, 1) # Reshape to be the correct size
    
    from random import randint
    for i in range(0, 10):
        plt.imshow(x_data[i*200 , :, :])
        plt.title(reverselookup[y_data[i*200 ,0]])
        plt.show()
    
    import keras
    from keras.utils import to_categorical
    y_data = to_categorical(y_data)
    
    
    x_data = x_data.reshape((datacount, 120, 320, 1))
    x_data /= 255
    return (x_data,y_data)


#from sklearn.model_selection import train_test_split
#x_train,x_further,y_train,y_further = train_test_split(x_data,y_data,test_size = 0.2)
#x_validate,x_test,y_validate,y_test = train_test_split(x_further,y_further,test_size = 0.5)
#
#
#from keras import layers
#from keras import models
#
#
#model=models.Sequential()
#model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(120, 320,1))) 
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Flatten())
#model.add(layers.Dense(128, activation='relu'))
#model.add(layers.Dense(10, activation='softmax'))
#
#model.compile(optimizer='rmsprop',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])
#model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1, validation_data=(x_validate, y_validate))
#model.save("mymodel.h5")
path="leapGestRecog/00/03_fist/frame_00_03_0001.png"
    
def predictImage(path):
    reverselookup={0: '01_palm',
     1: '02_l',
     2: '03_fist',
     3: '04_fist_moved',
     4: '05_thumb',
     5: '06_index',
     6: '07_ok',
     7: '08_palm_moved',
     8: '09_c',
     9: '10_down'}
    reverse_map = {0:'Fist',
                1:'L',
                2:'Okay',
                3:'Palm',
                4:'Peace'
                }
    from keras.models import load_model
    model = load_model('mymodel.h5')
    #x_data,y_data=prepareData()
    #[loss, acc] = model.evaluate(x_data,y_data,verbose=1)
    img = Image.open(path).convert('L')
                    # Read in and convert to greyscale
    img = img.resize((224, 224))
    img=np.array(img)
    ret, binary = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
    binary=binary.reshape(1,224,224,1)
    pred_array = model.predict(binary)
    result=reverse_map[np.argmax(pred_array[0])]
    return (result,pred_array[0])
#print("Accuracy:" + str(acc))
path="frames/thresh.jpg"
result,arr=predictImage(path)
print(result,arr)