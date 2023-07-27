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

match_resource = []
read_list = []

for i in dir_list:
    if '.jpg' in i:
        j = i.split("/")[-1].replace(".jpg", "")
        read_list.append(f'C:\\Users\\Richard\\Desktop\\ML-AI code bootcamp\\Final files\\Final image\\{i}')
        match_resource.append(int(j))

pl_csv['Image'] = np.NaN

for i in match_resource:
  for j in pl_csv['Resource ID(s)']:
    if i == j:
      index = match_resource.index(i)
      position = pl_csv.index[pl_csv['Resource ID(s)'] == j]
      pl_csv['Image'][position] = read_list[index]

cat_img_df = pl_csv.dropna()
cat_img_df = cat_img_df[['Mapped_Category', 'Image', 'Subject']]


y = cat_img_df['Mapped_Category']
X = cat_img_df['Image']
metadata_features = cat_img_df['Subject']
metadata_features = metadata_features.apply(preprocess)
metadata_features = list(metadata_features)

y = pd.get_dummies(y, drop_first = False)

def load_and_preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    return image

target_size = (224, 224)

X = np.array([load_and_preprocess_image(path, target_size) for path in X])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

vectorizer = CountVectorizer()

metadata_features = vectorizer.fit_transform(metadata_features).toarray()
image_input = Input(shape=(224, 224, 3))
metadata_input = Input(shape=(metadata_features.shape[1],))

conv1 = Conv2D(32, (3, 3), activation='relu')(image_input)
pool1 = MaxPooling2D((2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
pool2 = MaxPooling2D((2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu')(pool2)
pool3 = MaxPooling2D((2, 2))(conv3)

flatten_output = Flatten()(pool3)

concatenated_features = Concatenate()([flatten_output, metadata_input])


x = Dense(256, activation='relu')(concatenated_features)
predictions = Dense(len(category), activation='sigmoid')(x)


model = Model(inputs=[image_input, metadata_input], outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit([X_train, metadata_features[:X_train.shape[0]]], y_train, batch_size=32, epochs=10, validation_data=([X_test, metadata_features[X_train.shape[0]:]], y_test))

model.save('twelve_cat_CNN_extra.h5')
