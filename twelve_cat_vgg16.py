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
cat_img_df = cat_img_df[['Mapped_Category', 'Image']]


y = cat_img_df['Mapped_Category']
X = cat_img_df['Image']

y = pd.get_dummies(y, drop_first = False)

def load_and_preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    return image

target_size = (224, 224)

X = np.array([load_and_preprocess_image(path, target_size) for path in X])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)



base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
predictions = Dense(len(category), activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


X_testing, X_val, y_testing, y_val = train_test_split(X_test, y_test, test_size= 0.1, random_state= 2)

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

test_loss, test_accuracy = model.evaluate(X_testing, y_testing)


model.save('twelve_cat_vgg16_model.h5')