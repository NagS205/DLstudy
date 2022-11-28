import matplotlib.pyplot as plt
import os
import cv2
import random
import numpy as np
"""
DATADIR = "./dataset/kagglecatsanddogs_5340/PetImages/"
CATEGORIES = ["Dog", "Cat", "Dog_test", "Cat_test"]
IMG_SIZE = 80
training_data = []
"""
def create_training_data(DATADIR = "./dataset/kagglecatsanddogs_5340/PetImages/",
                        CATEGORIES = ["Dog", "Cat", "Dog_test", "Cat_test"],
                        IMG_SIZE = 80,
                        training_data = []):
    x_train, t_train, x_test, t_test = [], [], [], []
    
    for class_num, category in enumerate(CATEGORIES):
        path = os.path.join(DATADIR, category)
        for image_name in os.listdir(path)[:1000]:
            try:
                img_array = cv2.imread(os.path.join(path, image_name),)
                #image reading
                img_resize_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                #resize the image
                
                if class_num == 0:
                    x_train.append(img_resize_array)
                    t_train.append(0)
                elif class_num == 1:
                    x_train.append(img_resize_array)
                    t_train.append(1)
                elif class_num == 2:
                    x_test.append(img_resize_array)
                    t_test.append(0)
                elif class_num == 3:
                    x_test.append (img_resize_array)
                    t_test.append(1)
                # append data, label == 0 : dog, label == 1: cat    
                
                #training_data.append([img_resize_array, class_num])
                #append image data, label info
            except Exception as e:
                pass
        
    x_train = np.array(x_train)
    t_train = np.array(t_train)
    x_test = np.array(x_test)
    t_test = np.array(t_test)
    
    return (x_train, t_train), (x_test, t_test)

    
    
    # shuffle data
    # I realized later after writing this code that the following part was redundant.
    """
    random.shuffle(training_data)

    X_train = []
    Y_train = []

    for feature, label in training_data:
        X_train.append(feature)
        Y_train.append(label)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    # move data into arguments
    # following part should be first 
    x_train, t_train, x_test, t_test = [], [], [], []

    for i in range(len(X_train)):
        if Y_train[i] == 0:
            x_train.append(X_train[i])
            t_train.append(0)
        elif Y_train[i] == 1:
            x_train.append(X_train[i])
            t_train.append(1)
        elif Y_train[i] == 2:
            x_test.append(X_train[i])
            t_test.append(0)
        elif Y_train[i] == 3:
            x_test.append(X_train[i])
            t_test.append(1)

    x_train = np.array(x_train)
    t_train = np.array(t_train)
    x_test = np.array(x_test)
    t_test = np.array(t_test)

    return (x_train, t_train), (x_test, t_test)
    """