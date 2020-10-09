
from __future__ import print_function,division
from builtins import range,input


from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

import glob
import os
from os import getcwd
wd=getcwd()





#resize all the image to this
IMAGE_SIZE=[100,100]

#training config
epochs=5
batch_size= 32

train_path=wd+'\train'
valid_path=wd+'\test'


image_files=glob.glob(train_path + '/*/*.jp*g')
valid_image_files=glob.glob(valid_path + '/*/*.jp*g')


#useful for getting number of classes
folders=glob.glob(train_path+'/*')

#check image
plt.imshow(image.load_img(np.random.choice(image_files)))
plt.show()

#Add pre-processing layer to the front of the VGG
vgg=VGG16(input_shape=IMAGE_SIZE+[3],weights='imagenet',include_top=False)

#Dont train existing weights
for layer in vgg.layers:
    layer.trainable=False
    
#Our layer which we are attaching at the of VGG
x=Flatten()(vgg.output)
prediction=Dense(len(folders),activation='softmax')(x)



#Create a model object
model = Model(inputs=vgg.input,outputs=prediction)

#view the structure of the model
model.summary()


model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
)

#create an instance for ImageDataGenerator
gen=ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input)

test_gen=gen.flow_from_directory(valid_path,target_size=IMAGE_SIZE)
print(test_gen.class_indices)
labels=[None]*len(test_gen.class_indices )
for k,v in test_gen.class_indices.items():
    labels[v]=k

for x,y in test_gen:
    print("min:",x[0].min(),"max:",x[0].max())
    plt.title(labels[np.argmax(y[0])])
    plt.imshow(x[0])
    plt.show()
    break

#Create Generators
train_generator= gen.flow_from_directory(
        train_path,
        target_size=IMAGE_SIZE,
        shuffle=True,
        batch_size=batch_size)
valid_generator=gen.flow_from_directory(
        valid_path,
        target_size=IMAGE_SIZE,
        shuffle=True,
        batch_size=batch_size
)

#Fit the model
r=model.fit_generator(
        train_generator,
        validation_data=valid_generator,
        epochs=4,
        steps_per_epoch=len(image_files) // batch_size,
        validation_steps=len(valid_image_files) // batch_size
        )
model.save("model5.h5")

#Confusion Matrix
def get_confusion_matrix(data_path, N):
    print("Generating confusion matrix",N)
    predictions=[]
    targets=[]
    i=0
    for x,y in gen.flow_from_directory(data_path, target_size=IMAGE_SIZE,shuffle=False):
        i+=1
        if i%50 ==0:
            print(i)
        p=model.predict(x)
        p=np.argmax(p,axis=1)
        y=np.argmax(y,axis=1)
        predictions= np.concatenate((predictions,p))
        targets=np.concatenate((targets,y))
        if len(targets)>=N:
            break
    cm=confusion_matrix(targets,predictions)
    return cm
cm=get_confusion_matrix(train_path,len(image_files))
print(cm)
valid_cm=get_confusion_matrix(valid_path,len(valid_image_files))
print(valid_cm)

#Loss
plt.plot(r.history['loss'],label='train_loss')
plt.plot(r.history['val_loss'],label='val loss')
plt.legend()
plt.show()


#accuracy
plt.plot(r.history['accuracy'],label='train_acc')
plt.plot(r.history['val_accuracy'],label='val acc')
plt.legend()
plt.show()

































































































