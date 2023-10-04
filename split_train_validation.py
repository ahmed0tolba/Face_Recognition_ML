# run only once
# perfect 7 more than 40 images

import os, sys

set_folder =  'set1'
faces_folder = set_folder + '/Faces'
train_folder = set_folder + '/Training Images'
validation_folder = set_folder + '/Validation Images'
test_folder = set_folder + '/Testing Images'

faces_dirs  = os.listdir(faces_folder)
count = 0
for face_dir in faces_dirs:
    face_images = os.listdir(faces_folder + '/' + face_dir)
    if (len(face_images)>0): # if more than 1
        # create this face train dir if not exist 
        if not os.path.exists(train_folder + '/' + face_dir):
            os.makedirs(train_folder + '/' + face_dir)
        # then copy all images to train 
        for face_image in face_images:
            os.rename(faces_folder + '/' + face_dir + "/" + face_image , train_folder + '/' + face_dir + "/" + face_image)
        
    if (len(face_images)>0): # if more than 2
        # # create this face validate dir if not exist 
        # if not os.path.exists(validation_folder + '/' + face_dir):
        #     os.makedirs(validation_folder + '/' + face_dir)
        # # then copy first image from train to validate
        # os.rename(train_folder + '/' + face_dir + "/" + face_images[0] , validation_folder + '/' + face_dir + "/" + face_images[0])
        # create this face test dir if not exist 
        if not os.path.exists(test_folder + '/' + face_dir):
            os.makedirs(test_folder + '/' + face_dir)
        # then copy second image from train to validate
        os.rename(train_folder + '/' + face_dir + "/" + face_images[1] , test_folder + '/' + face_dir + "/" + face_images[1])

        
        count +=1
        # print(face_images)

print(count)                                   