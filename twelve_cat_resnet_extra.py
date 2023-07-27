from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import re
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Model
from keras.layers import Input, Concatenate
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

import re
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


path = 'C:\\Users\\Richard\\Desktop\\ML-AI code bootcamp\\Final files\\Final image'

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

pl_csv = pd.read_csv('C:\\Users\\Richard\\Desktop\\ML-AI code bootcamp\\Final files\\Final image\\events_edited.csv', index_col=0)

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

resNet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in resNet_model.layers:
    layer.trainable = False

resnet_output = resNet_model(image_input)

flatten_output = Flatten()(resnet_output)

concatenated_features = Concatenate()([flatten_output, metadata_input])

x = Dense(512, activation='relu')(concatenated_features)
predictions = Dense(len(category), activation='sigmoid')(x)

model = Model(inputs=[image_input, metadata_input], outputs=predictions)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit([X_train, metadata_features[:X_train.shape[0]]], y_train, batch_size=32, epochs=10, validation_data=([X_test, metadata_features[X_train.shape[0]:]], y_test))


model.save('twelve_cat_resnet_extra.h5')