#importing necessary libraries
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Flatten, Dense
from keras.applications import VGG16
from keras.optimizers import Adam
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

#Path to file with images and metadata csv
path = 'C:\\Users\\Richard\\Desktop\\ML-AI code bootcamp\\Final files\\Final image'

#categories to sort by
category = ["Africana Studies",
"American Studies",
"Arabic Studies",
"Asian Studies Program",
"Classics Greek  Latin",
"East Asian Languages  Cultures",
'Environmental Studies',
"History",
"Religion",
"Science Technology Studies",
"Sociology",
"Women Gender Sexuality Studies"]

#reading in the csv from the path
pl_csv = pd.read_csv('C:\\Users\\Richard\\Desktop\\ML-AI code bootcamp\\Final files\\Final image\\events_edited.csv', index_col=0)
#setting the path of the directory with the files
dir_list = os.listdir(path)

#Finding and mapping each image in the file path to the image labels
match_resource = []
read_list = []

for i in dir_list:
    if '.jpg' in i:
        j = i.split("/")[-1].replace(".jpg", "")
        read_list.append(f'C:\\Users\\Richard\\Desktop\\ML-AI code bootcamp\\Final files\\Final image\\{i}')
        match_resource.append(int(j))

#Updating the dataframe so it containes the image paths connected with each label
pl_csv['Image'] = np.NaN

for i in match_resource:
  for j in pl_csv['Resource ID(s)']:
    if i == j:
      index = match_resource.index(i)
      position = pl_csv.index[pl_csv['Resource ID(s)'] == j]
      pl_csv['Image'][position] = read_list[index]

cat_img_df = pl_csv.dropna()
cat_img_df = cat_img_df[['Mapped_Category', 'Image']]

#Optaining the X(image) and Y(label) for each image
y = cat_img_df['Mapped_Category']
X = cat_img_df['Image']

#Vectorizing the labels
y = pd.get_dummies(y, drop_first = False)

#function to process the images to the right size
def load_and_preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    return image

#The target size
target_size = (224, 224)

#fitting all the images in X to the right size
X = np.array([load_and_preprocess_image(path, target_size) for path in X])

#Getting the train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

#setting up the VGG-16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#Freezing some layers so they won't change during training
for layer in base_model.layers:
    layer.trainable = False

#Using the VGG-16 model
#getting and flattening the output of the VGG-16
x = Flatten()(base_model.output)
#Dense layer
x = Dense(256, activation='relu')(x)
#Dense layer predicting outputs
predictions = Dense(len(category), activation='sigmoid')(x)

#Setting up the model
model = Model(inputs=base_model.input, outputs=predictions)

#Compiling the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

#Fitting the model on the training and testing images
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

#saving the model
model.save('twelve_cat_vgg16_model.h5')
