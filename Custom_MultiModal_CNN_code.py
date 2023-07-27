#importing necessary libraries
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import numpy as np
import re
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Model
from keras.layers import Input, Concatenate

#function to preprocess text
def preprocess(text):
  text = str(text)
  text = text.lower()
  text = re.sub(r"what's", "what is ", text)
  text = re.sub(r"\'s", " ", text)
  text = re.sub(r"\'ve", " have ", text)
  text = re.sub(r"can't", "can not ", text)
  text = re.sub(r"n't", " not ", text)
  text = re.sub(r"i'm", "i am ", text)
  text = re.sub(r"\'re", " are ", text)
  text = re.sub(r"\'d", " would ", text)
  text = re.sub(r"\'ll", " will ", text)
  text = re.sub(r"\'scuse", " excuse ", text)
  text = re.sub('\W', ' ', text)
  text = re.sub('\s+', ' ', text)
  text = text.strip(' ')
  return text

#path to the file with all the images and dataset
path = 'C:\\Users\\Richard\\Desktop\\ML-AI code bootcamp\\Final files\\Final image'

#The categories the model trains on
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

#reading in the dataset
pl_csv = pd.read_csv('C:\\Users\\Richard\\Desktop\\ML-AI code bootcamp\\Final files\\Final image\\events_edited.csv', index_col=0)

#setting the path to the directory with all the necessary files
dir_list = os.listdir(path)

#Finding and mapping each image in the file path to the labels (images are named by resourceID)
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
cat_img_df = cat_img_df[['Mapped_Category', 'Image', 'Subject']]

#Optaining the X(image) and Y(label) for each image
y = cat_img_df['Mapped_Category']
X = cat_img_df['Image']

#Getting the additional features (keywords) from the dataset
metadata_features = cat_img_df['Subject']
metadata_features = metadata_features.apply(preprocess)
metadata_features = list(metadata_features)

#Vectorizing the labels
y = pd.get_dummies(y, drop_first = False)

#function to process the images to the right size
def load_and_preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    return image

#The target image size
target_size = (224, 224)

#fitting all the images in X to the right size
X = np.array([load_and_preprocess_image(path, target_size) for path in X])

#Getting the train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

#Loading the vectorizer
vectorizer = CountVectorizer()

#Vectorizing the additional features
metadata_features = vectorizer.fit_transform(metadata_features).toarray()

#Setting up the image and additional features input
image_input = Input(shape=(224, 224, 3))
metadata_input = Input(shape=(metadata_features.shape[1],))

#Creating the CNN model taking in the image input
#Convolution-relu, maxpooling layers
conv1 = Conv2D(32, (3, 3), activation='relu')(image_input)
pool1 = MaxPooling2D((2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
pool2 = MaxPooling2D((2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu')(pool2)
pool3 = MaxPooling2D((2, 2))(conv3)

#flattenining the Convolution layers
flatten_output = Flatten()(pool3)

#adding in the additional features to the image features
concatenated_features = Concatenate()([flatten_output, metadata_input])

#dense layer with all the features
x = Dense(256, activation='relu')(concatenated_features)
#dense layer outputting to prediction categories
predictions = Dense(len(category), activation='sigmoid')(x)

#Setting up the model
model = Model(inputs=[image_input, metadata_input], outputs=predictions)
#Compiling the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
#Fitting the model on the images, labels, and additional features
model.fit([X_train, metadata_features[:X_train.shape[0]]], y_train, batch_size=32, epochs=10, validation_data=([X_test, metadata_features[X_train.shape[0]:]], y_test))

#saving the models
model.save('twelve_cat_CNN_extra.h5')
