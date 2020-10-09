import numpy as np, math
import cv2
import matplotlib.pyplot as plt
import cv2.imshow as imshow
import keras
import skimage.morphology
 
import matplotlib.pyplot as plt
 
model = keras.models.load_model('model3.h5')
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model_labels = {0:"diode", 1:"resistor", 2:"inductor", 3:"ground", 4:"voltage", 5:"cap"}
 
class Circuit():
  def __init__(self, image_path):
 
    self.start_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) #don't touch this.
 
    if self.start_image.shape[-1] == 4: #if it has alpha channel
      trans_mask = self.start_image[:,:,3] == 0 #mask wherever alpha is 0
      self.start_image[trans_mask] = [255, 255, 255, 255]
      self.start_image = cv2.cvtColor(self.start_image, cv2.COLOR_BGRA2BGR)
 
    self.width = self.start_image.shape[0] 
    self.height = self.start_image.shape[1]
    self.area = self.height * self.width

    self.grayscaled = cv2.cvtColor(self.start_image.copy(), cv2.COLOR_BGR2GRAY)
    self.grayscaled = cv2.resize(self.grayscaled, (self.height,self.width))
    _, self.thresholded = cv2.threshold(self.grayscaled, 40, 255, cv2.THRESH_BINARY)
    self.skeletonized = (skimage.morphology.skeletonize(self.thresholded < 100) * 255).astype('uint8')
    self.pipelined = self.skeletonized.copy()
 
    self.lines = []
    self.components = []
    self.extracted_components = []
    self.blank = self.pipelined * 0

    self.remove_lines()
    self.find_components(reject_below_factor=0).find_components(True,False,0.005, True)
    self.predict()

  def remove_lines(self, thickness=10):
    self.canny = cv2.Canny(self.pipelined, 100,200)
    self.hline_thresh = 75
    self.hline_min_line_length = min(self.height, self.width)*0.1
    self.hline_max_line_gap = min(self.height, self.width)*0.1
    self.all_lines = cv2.HoughLinesP(self.canny, 1, np.pi/180, 
                           self.hline_thresh, None, 
                           self.hline_min_line_length, self.hline_max_line_gap) #TODO
    for i,line in enumerate(self.all_lines):
      for x1,y1,x2,y2 in line:
        length= math.sqrt((x1-x2)**2 + (y1-y2)**2)
        self.lines.append({"label":f"line_{i}", "coordinates":(x1,y1,x2,y2), "length":length})
        cv2.line(self.pipelined,(x1,y1),(x2,y2),0,thickness)
    return self
 
  def find_components(self, draw_on_blank=False, draw_on_pipeline=True, reject_below_factor=0.005, store=False):
    """Find contours. Blob it up."""
    self.blank = self.pipelined * 0
    acceptable_area = (0, self.area * 0.8)
    thickness = 25
    all_contours, h1 = cv2.findContours(self.pipelined,cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
    self.contours = [cnt for cnt in all_contours if acceptable_area[0] <= cv2.contourArea(cnt) < acceptable_area[1]]
    for cnt in self.contours:
      x,y,w,h = cv2.boundingRect(cnt)
      if w*h > self.area * reject_below_factor:
        if store: self.components.append({"loc":(x,y),"dim":(w,h), "img":self.start_image[y:y+h, x:x+w]})
        if draw_on_blank:cv2.rectangle(self.blank,(x,y),(x+w,y+h),(123,0,12),10)
        if draw_on_pipeline:cv2.drawContours(self.pipelined, cnt, -1, 255, thickness)
    return self

  def predict(self):
    for component in self.components:
      img = cv2.resize(component["img"],(100,100), 3)
      img = np.reshape(img, (1,100,100,3))
      classes = list(model.predict(img)[0])
      component["label"]=model_labels[classes.index(max(classes))]

  def show_with_labels(self):
    for component in self.components:
      cv2.imshow(component["img"])
      print(component["label"])

ckt0 = Circuit("/content/drive/My Drive/colab_images/res.png")
ckt1 = Circuit("/content/drive/My Drive/colab_images/hand.png")
ckt2 = Circuit("/content/drive/My Drive/colab_images/nolabel.png")
ckt3 = Circuit("/content/drive/My Drive/colab_images/rlc.png")
test_list = [ckt0, ckt1, ckt2, ckt3]

ckt0.show_with_labels()
ckt1.show_with_labels()
ckt2.show_with_labels()







For analysis of the proposed  method ,a total of 5 circuit images were used comprising hand drawn images. These circuits comprise of 5 components which are present in the data set. The data set consists of 100 images for each component. For segmentation almost all components could be identified in the circuit image and could be extracted. An accuracy of 80% was achieved in segmenting and extracting the components.
In our first approach with recognition of the segmented components, we used SVM classifiers to classify these images to their respective labels. In this classification about 400 images were used, which comprised of our own dataset of hand drawn components. On running a test set comprising of 80 images, we achieved an accuracy of 56%.


With our second approach we used the same procedure to process the image and used CNN to identify the components in the same. The same dataset was used which comprised of 400 images (80 images for each component). We compiled the whole CNN model for 50 epochs and achieved an accuracy of 83% in the detecting the components in the image.

















