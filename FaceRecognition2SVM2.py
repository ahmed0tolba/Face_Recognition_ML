import numpy as np 
import pandas as pd 
import cv2
import time
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.feature import hog # pip install scikit-image
import os
import re
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report , accuracy_score

set_folder =  'set5'
train_folder = set_folder + '/Training Images'
test_folder = set_folder + '/Testing Images'

t1=time.time()
height=64
width=64
train_data=[]
train_labels=[]
test_data=[]
test_labels=[]
Celebs=[]

for dirname,_, train_filenames in tqdm(os.walk(train_folder)):
    for filename in train_filenames:
        image = cv2.imread(os.path.join(dirname, filename))
        image= cv2.resize(image , (width,height))
        train_labels.append(dirname.split("/")[-1])
        train_data.append(image)

for dirname,_, test_filenames in tqdm(os.walk(test_folder)):
    for filename in test_filenames:
        image = cv2.imread(os.path.join(dirname, filename))
        image= cv2.resize(image , (width,height))
        test_labels.append(dirname.split("/")[-1])
        test_data.append(image)

# fig = plt.figure(figsize=(20,15))
# for i in range(1,10):
#     index = random.randint(0,len(train_filenames)-1)
#     plt.subplot(3,3,i)
#     plt.imshow(train_data[index])
#     plt.xlabel(train_labels[index])
# plt.show()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_Labels= le.fit_transform(train_labels)
test_Labels= le.fit_transform(test_labels)

train_data_gray = [cv2.cvtColor(train_data[i] , cv2.COLOR_BGR2GRAY) for i in range(len(train_data))]
test_data_gray = [cv2.cvtColor(test_data[i] , cv2.COLOR_BGR2GRAY) for i in range(len(test_data))]

# fig = plt.figure(figsize=(20,15))
# for i in range(1,10):
#     index = random.randint(1,len(train_filenames)-1)
#     plt.subplot(3,3,i)
#     plt.imshow(train_data_gray[index])
#     plt.xlabel(train_Labels[index])
# plt.show()

train_Labels = np.array(train_Labels).reshape(len(train_Labels),1)
test_Labels = np.array(test_Labels).reshape(len(test_Labels),1)

ppc =8
cb=4
train_hog_features=[]
train_hog_image=[]
for image in tqdm(train_data_gray):
    fd , hogim = hog(image , orientations=9 , pixels_per_cell=(ppc , ppc) , block_norm='L2' , cells_per_block=(cb,cb) , visualize=True)
    train_hog_image.append(hogim)
    train_hog_features.append(fd)
test_hog_features=[]
test_hog_image=[]
for image in tqdm(test_data_gray):
    fd , hogim = hog(image , orientations=9 , pixels_per_cell=(ppc , ppc) , block_norm='L2' , cells_per_block=(cb,cb) , visualize=True)
    test_hog_image.append(hogim)
    test_hog_features.append(fd)

# fig = plt.figure(figsize=(20,15))
# for i in range(1,10):
#     index = random.randint(1,len(train_filenames)-1)
#     plt.subplot(3,3,i)
#     plt.imshow(train_hog_image[index])
#     plt.xlabel(train_Labels[index])
# plt.show()

train_hog_features = np.array(train_hog_features)
train_df = np.hstack((train_hog_features,train_Labels))

test_hog_features = np.array(test_hog_features)
test_df = np.hstack((test_hog_features,test_Labels))

# X_train , X_test  = train_df[:,:-1] , train_df[:,-1]
# Y_train , Y_test = test_df[:,:-1] , test_df[:,-1]
X_train , Y_train = train_df[:,:-1] , train_df[:,-1]
X_test , Y_test = test_df[:,:-1] , test_df[:,-1]

from sklearn.decomposition import PCA
t= time.time()
pca = PCA(n_components=150 , svd_solver='randomized' , whiten=True).fit(X_train)
print("Time evolved", time.time()-t)

print("Projecting the input data on the orthonormal basis")
t0 = time.time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
# X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time.time() - t0))

t3=time.time()
svm = SVC(kernel='rbf' , class_weight='balanced' , C=1000 , gamma=0.0082)
svm.fit(X_train_pca , Y_train)
print("score" , svm.score(X_test_pca , Y_test))
print("done in %0.3fs" % (time.time() - t3))

print("total time evolved", (time.time()-t1))