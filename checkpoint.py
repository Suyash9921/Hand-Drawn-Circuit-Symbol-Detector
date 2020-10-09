import keras

from keras.models import load_model
import cv2
import numpy as np

model = load_model('model4.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
label=['Diode','Resistor','Inductor','Ground','Voltage','Capacitor']
img = cv2.imread('ind.jpeg')
img = cv2.resize(img,(100,100))
img = np.reshape(img,[1,100,100,3])

classes = model.predict(img)
lis=(classes[0]).tolist()
k=lis.index(max(lis))
print(label[k])
