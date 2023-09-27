import tensorflow as tf
import keras
import keras.utils as image
import numpy as np

test_folder = 'set3/Testing Images/'

model_cnn = keras.models.load_model("model_set3.keras")

# Testing folder

ImagePath = test_folder + 'face4/3face4.jpg'
test_image=image.load_img(ImagePath,target_size=(64, 64))
test_image=image.img_to_array(test_image)
 
test_image=np.expand_dims(test_image,axis=0)
 
result=model_cnn.predict(test_image,verbose=0)
#print(training_set.class_indices)
 
print('####'*10)
print('Prediction is: ',ResultMap[np.argmax(result)])