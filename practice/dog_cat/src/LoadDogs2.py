import matplotlib.pyplot as plt
import os
import cv2
import random
import numpy as np
import tensorflow as tf
# library for data augmentation
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
array_to_img = tf.keras.utils.array_to_img
img_to_array = tf.keras.utils.img_to_array
"""
DATADIR = "./dataset/kagglecatsanddogs_5340/PetImages/"
CATEGORIES = ["Dog", "Cat", "Dog_test", "Cat_test"]
IMG_SIZE = 56
training_data = []
"""
def create_training_data(DATADIR = "./dataset/kagglecatsanddogs_5340/PetImages/",
                        CATEGORIES = ["Dog", "Dog_test", "Cat", "Cat_test",
                                      "rhinoceros","rhinoceros_test", "stag", "stag_test"],
                        IMG_SIZE = 56,
                        training_data = []):
    img_train, label_train, img_test, label_test = [], [], [], []
    
    #set datagen function
    datagen = ImageDataGenerator(rotation_range=180, zoom_range=[0.9,1.1],
                                horizontal_flip=True, vertical_flip=True, brightness_range = [0.5, 0.8])
    
    for class_num, category in enumerate(CATEGORIES):
        path = os.path.join(DATADIR, category)
        for image_name in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, image_name),)
                #image reading
                #Dog
                if class_num == 0:
                    img_resize_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    #resize the image
                    img_train.append(img_resize_array)
                    label_train.append(0)
                #Dog_test
                elif class_num == 1:
                    img_resize_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    #resize the image
                    img_test.append(img_resize_array)
                    label_test.append(0)
                #Cat
                elif class_num == 2:
                    img_resize_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    #resize the image
                    img_train.append(img_resize_array)
                    label_train.append(1)
                #Cat_test
                elif class_num == 3:
                    img_resize_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    #resize the image
                    img_test.append(img_resize_array)
                    label_test.append(1)
                #rhinoceros
                elif class_num == 4:
                    img_rhinoceros = img_array[np.newaxis, :, : ,:]
                    for i, data in enumerate(datagen.flow(img_rhinoceros, batch_size = 1, seed = random.randint(1,10000))):
                        img_resize_array = cv2.resize(data[0], (IMG_SIZE, IMG_SIZE))
                        img_train.append(img_resize_array)
                        label_train.append(2)
                        if i == 250:
                            break
                            
                #rhinoceros_test
                elif class_num == 5:
                    img_rhinoceros = img_array[np.newaxis, :, : ,:]
                    for i, data in enumerate(datagen.flow(img_rhinoceros, batch_size = 1, seed = random.randint(1,10000))):
                        img_resize_array = cv2.resize(data[0], (IMG_SIZE, IMG_SIZE))
                        img_test.append(img_resize_array)
                        label_test.append(2)
                        #print("augumentation")
                        if i == 100:
                            break
                            
                #stag
                elif class_num == 6:
                    img_stag = img_array[np.newaxis, :, : ,:]
                    for i, data in enumerate(datagen.flow(img_stag, batch_size = 1, seed = random.randint(1,10000))):
                        img_resize_array = cv2.resize(data[0], (IMG_SIZE, IMG_SIZE))
                        img_train.append(img_resize_array)
                        label_train.append(3)
                        if i == 250:
                            break
                            
                #stag_test
                elif class_num == 7:
                    img_stag = img_array[np.newaxis, :, : ,:]
                    for i, data in enumerate(datagen.flow(img_stag, batch_size = 1, seed = random.randint(1,10000))):
                        img_resize_array = cv2.resize(data[0], (IMG_SIZE, IMG_SIZE))
                        img_test.append(img_resize_array)
                        label_test.append(3)
                        #print("augumentation")
                        if i == 100:
                            break
                            
                
                
                # append data, label == 0 : dog, label == 1: cat, label == 2 : rhinoceros    
                
                #training_data.append([img_resize_array, class_num])
                #append image data, label info
            except Exception as e:
                pass
        
    img_train = np.array(img_train)
    label_train = np.array(label_train)
    img_test = np.array(img_test)
    label_test = np.array(label_test)
    
    return (img_train, label_train), (img_test, label_test)