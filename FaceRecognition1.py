from keras.preprocessing.image import ImageDataGenerator
import pickle
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
import time


set_folder =  'set5'
train_folder = set_folder + '/Training Images'

# TestingImagePath = TrainingImagePath
train_gen = ImageDataGenerator(shear_range=0.1,zoom_range=0.1,horizontal_flip=True)
# test_gen = ImageDataGenerator()
 
training_data = train_gen.flow_from_directory(train_folder,target_size=(128, 128),batch_size=32,class_mode='categorical')
# test_data = test_gen.flow_from_directory(test_folder,target_size=(64, 64),batch_size=32,class_mode='categorical')
 
# print(training_data.class_indices)

#Creating lookup table for faces
# class_indices have the numeric tag for each face
TrainClasses = training_data.class_indices
 
# Storing the face and the numeric tag for future reference
ResultMap={}
for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
    ResultMap[faceValue]=faceName
 
# Saving the face map for future reference
with open("ResultsMap.pkl", 'wb') as fileWriteStream:
    pickle.dump(ResultMap, fileWriteStream)
 
# The model will give answer as a numeric tag
# This mapping will help to get the corresponding face name for it
# print("Mapping of Face and its ID",ResultMap)
 
# The number of neurons for the output layer is equal to the number of faces
OutputNeurons=len(ResultMap)
print('\n The Number of output neurons: ', OutputNeurons)

model_cnn = Sequential()
model_cnn.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(128,128,3), activation='relu'))# 1 Convolution Layer
model_cnn.add(MaxPool2D(pool_size=(2,2)))# 2 MAX Pooling
model_cnn.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))# 3 Convolution Layer 
model_cnn.add(MaxPool2D(pool_size=(2,2)))# 4 MAX Pooling
model_cnn.add(Convolution2D(128, kernel_size=(5, 5), strides=(1, 1), activation='relu'))# Convolution Layer 
model_cnn.add(MaxPool2D(pool_size=(2,2)))# MAX Pooling
model_cnn.add(Flatten())# FLattening
model_cnn.add(Dense(128, activation='relu'))# Dense
model_cnn.add(Dense(OutputNeurons, activation='softmax'))# Output layer Dense , softmax because it is classification output
model_cnn.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])# Creating a trainer 

# saving model
model_cnn.save(set_folder + ".keras")

# Measuring the time taken by the model to train
StartTime=time.time()
 
# Starting the model training
model_cnn.fit(
                    training_data,
                    # steps_per_epoch=30,
                    epochs=100,
                    # validation_data=test_data,
                    validation_steps=10
                    )
 
EndTime=time.time()
print("Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ')


# not learning
# model_cnn.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), input_shape=(128,128,3), activation='relu'))# 1 Convolution Layer
# model_cnn.add(MaxPool2D(pool_size=(2,2)))# 2 MAX Pooling
# model_cnn.add(Convolution2D(128, kernel_size=(5, 5), strides=(1, 1), activation='relu'))# 3 Convolution Layer 
# model_cnn.add(MaxPool2D(pool_size=(2,2)))# 4 MAX Pooling
# model_cnn.add(Flatten())# 5 FLattening
# model_cnn.add(Dense(128, activation='relu'))# 6 Dense
# model_cnn.add(Dense(OutputNeurons, activation='softmax'))# Output layer Dense , softmax because it is classification output
# model_cnn.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])# Creating a trainer 
# Epoch 17/100
# 191/191 [==============================] - 354s 2s/step - loss: 6.8042 - accuracy: 0.0062

## 51
# 1 Convolution Layer 
# model_cnn.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(128,128,3), activation='relu'))
# 2 MAX Pooling
# model_cnn.add(MaxPool2D(pool_size=(2,2)))
# 3 Convolution Layer 
# model_cnn.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
# 4 MAX Pooling
# model_cnn.add(MaxPool2D(pool_size=(2,2)))
# 5 FLattening
# model_cnn.add(Flatten())
# 6 Dense
# model_cnn.add(Dense(128, activation='relu'))
# Output layer Dense , softmax because it is classification output
# model_cnn.add(Dense(OutputNeurons, activation='softmax'))
# Creating a trainer 
# model_cnn.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])
# Epoch 100/100 
# 191/191 [==============================] - 140s 731ms/step - loss: 0.8168 - accuracy: 0.8115
# Total Time Taken:  244 Minutes