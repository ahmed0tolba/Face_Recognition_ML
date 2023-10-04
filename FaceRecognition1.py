from keras.preprocessing.image import ImageDataGenerator
import pickle
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
import time
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

set_folder =  'set5'
train_folder = set_folder + '/Training Images'
# validation_folder = set_folder + '/Validation Images'

# TestingImagePath = TrainingImagePath
train_gen = ImageDataGenerator(shear_range=0.1,zoom_range=0.1,horizontal_flip=True)
# test_gen = ImageDataGenerator()
 
training_data = train_gen.flow_from_directory(train_folder,target_size=(64, 64),batch_size=32,class_mode='categorical')
# validation_data = test_gen.flow_from_directory(validation_folder,target_size=(128, 128),batch_size=32,class_mode='categorical')
 
# print(training_data.class_indices)

#Creating lookup table for faces
# class_indices have the numeric tag for each face
TrainClasses = training_data.class_indices
 
# Storing the face and the numeric tag for future reference
ResultMap={}
for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
    ResultMap[faceValue]=faceName
 
# Saving the face map for future reference
with open("ResultsMapset.pkl", 'wb') as fileWriteStream:
    pickle.dump(ResultMap, fileWriteStream)
 
# The model will give answer as a numeric tag
# This mapping will help to get the corresponding face name for it
# print("Mapping of Face and its ID",ResultMap)
 
# The number of neurons for the output layer is equal to the number of faces
OutputNeurons=len(ResultMap)
print('\n The Number of output neurons: ', OutputNeurons)

model_cnn = Sequential()
model_cnn.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64,64,3), activation='relu'))# 1 Convolution Layer
model_cnn.add(MaxPool2D(pool_size=(2,2)))# 2 MAX Pooling
model_cnn.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))# 3 Convolution Layer 
model_cnn.add(MaxPool2D(pool_size=(2,2)))# 4 MAX Pooling
# model_cnn.add(Convolution2D(128, kernel_size=(5, 5), strides=(1, 1), activation='relu'))# Convolution Layer 
# model_cnn.add(MaxPool2D(pool_size=(2,2)))# MAX Pooling
model_cnn.add(Flatten())# FLattening
model_cnn.add(Dense(2048, activation='relu')) # Dense
model_cnn.add(Dense(OutputNeurons, activation='softmax')) # Output layer Dense , softmax because it is classification output
model_cnn.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"]) # Creating a trainer 



# Measuring the time taken by the model to train
StartTime=time.time()

earlystop = EarlyStopping(monitor='val_accuracy',mode="max",patience=50, restore_best_weights=True)

# Starting the model training
history = model_cnn.fit(training_data,steps_per_epoch=30,epochs=200,validation_data=training_data,validation_steps=10,callbacks=[earlystop])
# saving model
model_cnn.save("model" + set_folder + ".keras")

EndTime=time.time()
print("Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ')

print(history.history.keys())

plt.figure(1)
# summarize history for accuracy
plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')

# summarize history for loss
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

# set5 selective test 
# model_cnn.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64,64,3), activation='relu'))# 1 Convolution Layer
# model_cnn.add(MaxPool2D(pool_size=(2,2)))# 2 MAX Pooling
# model_cnn.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))# 3 Convolution Layer 
# model_cnn.add(MaxPool2D(pool_size=(2,2)))# 4 MAX Pooling
# model_cnn.add(Convolution2D(128, kernel_size=(5, 5), strides=(1, 1), activation='relu'))# Convolution Layer 
# model_cnn.add(MaxPool2D(pool_size=(2,2)))# MAX Pooling
# model_cnn.add(Flatten())# FLattening
# model_cnn.add(Dense(1024, activation='relu')) # Dense
# model_cnn.add(Dense(OutputNeurons, activation='softmax')) # Output layer Dense , softmax because it is classification output
# model_cnn.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"]) # Creating a trainer 
# Epoch 100/100
# 30/30 [==============================] - 7s 222ms/step - loss: 3.5388 - accuracy: 0.2521 - val_loss: 3.5282 - val_accuracy: 0.2688
# Total Time Taken:  11 Minutes

# set5 selective test 18 23 0.782608695652174  - 100%
# model_cnn.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64,64,3), activation='relu'))# 1 Convolution Layer
# model_cnn.add(MaxPool2D(pool_size=(2,2)))# 2 MAX Pooling
# model_cnn.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))# 3 Convolution Layer 
# model_cnn.add(MaxPool2D(pool_size=(2,2)))# 4 MAX Pooling
# # model_cnn.add(Convolution2D(128, kernel_size=(5, 5), strides=(1, 1), activation='relu'))# Convolution Layer 
# # model_cnn.add(MaxPool2D(pool_size=(2,2)))# MAX Pooling
# model_cnn.add(Flatten())# FLattening
# model_cnn.add(Dense(1024, activation='relu')) # Dense
# model_cnn.add(Dense(OutputNeurons, activation='softmax')) # Output layer Dense , softmax because it is classification output
# model_cnn.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"]) # Creating a trainer 
# history = model_cnn.fit(training_data,steps_per_epoch=30,epochs=100,validation_data=training_data,validation_steps=10,callbacks=[earlystop])
# Epoch 100/100
# 30/30 [==============================] - 8s 254ms/step - loss: 1.0734 - accuracy: 0.7688 - val_loss: 0.8689 - val_accuracy: 0.7844
# Total Time Taken:  13 Minutes

# Set5 100 epochs max val_accuracy = 31 % , val images are not related or overfitting (not over fitting because it succeded with set1 which has less users)
# model_cnn.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(128,128,3), activation='relu'))# 1 Convolution Layer
# model_cnn.add(MaxPool2D(pool_size=(2,2)))# 2 MAX Pooling
# model_cnn.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))# 3 Convolution Layer 
# model_cnn.add(MaxPool2D(pool_size=(2,2)))# 4 MAX Pooling
# # model_cnn.add(Convolution2D(128, kernel_size=(5, 5), strides=(1, 1), activation='relu'))# Convolution Layer 
# # model_cnn.add(MaxPool2D(pool_size=(2,2)))# MAX Pooling
# model_cnn.add(Flatten())# FLattening
# model_cnn.add(Dense(64, activation='relu')) # Dense
# model_cnn.add(Dense(OutputNeurons, activation='softmax')) # Output layer Dense , softmax because it is classification output
# model_cnn.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"]) # Creating a trainer 
# Epoch 100/100
# 62/62 [==============================] - 53s 850ms/step - loss: 0.2247 - accuracy: 0.9384 - val_loss: 15.2145 - val_accuracy: 0.0938
# Total Time Taken:  78 Minutes

# Set1 20 epoch perfect
# model_cnn.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(128,128,3), activation='relu'))# 1 Convolution Layer
# model_cnn.add(MaxPool2D(pool_size=(2,2)))# 2 MAX Pooling
# model_cnn.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))# 3 Convolution Layer 
# model_cnn.add(MaxPool2D(pool_size=(2,2)))# 4 MAX Pooling
# # model_cnn.add(Convolution2D(128, kernel_size=(5, 5), strides=(1, 1), activation='relu'))# Convolution Layer 
# # model_cnn.add(MaxPool2D(pool_size=(2,2)))# MAX Pooling
# model_cnn.add(Flatten())# FLattening
# model_cnn.add(Dense(64, activation='relu')) # Dense
# model_cnn.add(Dense(OutputNeurons, activation='softmax')) # Output layer Dense , softmax because it is classification output
# model_cnn.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"]) # Creating a trainer 

# Set1 20 epoch early exit
# model_cnn.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(128,128,3), activation='relu'))# 1 Convolution Layer
# model_cnn.add(MaxPool2D(pool_size=(2,2)))# 2 MAX Pooling
# model_cnn.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))# 3 Convolution Layer 
# model_cnn.add(MaxPool2D(pool_size=(2,2)))# 4 MAX Pooling
# # model_cnn.add(Convolution2D(128, kernel_size=(5, 5), strides=(1, 1), activation='relu'))# Convolution Layer 
# # model_cnn.add(MaxPool2D(pool_size=(2,2)))# MAX Pooling
# model_cnn.add(Flatten())# FLattening
# model_cnn.add(Dense(128, activation='relu')) # Dense
# model_cnn.add(Dense(OutputNeurons, activation='softmax')) # Output layer Dense , softmax because it is classification output
# model_cnn.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"]) # Creating a trainer 
# Epoch 20/20
# 9/9 [==============================] - 7s 779ms/step - loss: 1.5422 - accuracy: 0.5725 - val_loss: 2.0059 - val_accuracy: 0.5000
# Total Time Taken:  2 Minutes