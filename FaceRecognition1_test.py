import tensorflow as tf
import keras
import keras.utils as image
import numpy as np
import pickle
import os, sys

with open('ResultsMapset.pkl', 'rb') as f:
    ResultMap = pickle.load(f)

test_folder = 'set5/Testing Images/'

model_cnn = keras.models.load_model("modelset5.keras")

c= 0
cg = 0
faces_dirs  = os.listdir(test_folder)
for face_dir in faces_dirs:
    images = os.listdir(test_folder + "/" + face_dir)
    for faceimage in images:
        ImagePath = test_folder + "/" + face_dir + '/' + faceimage
        test_image=image.load_img(ImagePath,target_size=(64, 64))
        test_image=image.img_to_array(test_image)
        test_image=np.expand_dims(test_image,axis=0)
        result=model_cnn.predict(test_image,verbose=0)
        #print(training_set.class_indices)
        print(face_dir ,np.argmax(result), 'Prediction is: ',ResultMap[np.argmax(result)])
        c +=1
        if face_dir == ResultMap[np.argmax(result)]:
            cg +=1

print(cg , c,cg / c)