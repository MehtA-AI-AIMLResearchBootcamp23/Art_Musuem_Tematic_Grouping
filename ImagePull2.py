from PIL import Image
import requests
from io import BytesIO
import pandas as pd

import os
path = 'C:\\Users\\Richard\\Desktop\\ML-AI code bootcamp\\Final files\\Final image'

dir_list = os.listdir(path)

read_list = []

for i in dir_list:
    if '.jpg' in i:
        i = i.split("/")[-1].replace(".jpg", "")
        read_list.append(int(i))

ee_nn = pd.read_csv(r'C:\Users\Richard\Desktop\ML-AI code bootcamp\Final files\Final image\metadata.csv')

resourceID = ee_nn['Resource ID(s)_x']
resourceID = list(set(resourceID))
resourceID = [i for i in resourceID if i not in read_list]
print(len(resourceID))

for id in resourceID:
  myID = id
  thumbnail_url = f"https://rs.williams.edu/iiif/image/{myID}/full/150,/0/default.jpg"

  response = requests.get(thumbnail_url)

  if response.status_code == 200:
  # Extract the file name from the URL
    filename = f"{myID}.jpg"

    # Save the image to a file
    with open(filename, "wb") as f:
      f.write(response.content)

    print(f"Image downloaded and saved as '{filename}'.")
  else:
    print("Failed to fetch the image.")


print('finished all labeled image')


path = 'C:\\Users\\Richard\\Desktop\\ML-AI code bootcamp\\Final files\\Final image'

dir_list = os.listdir(path)

read_list = []

for i in dir_list:
    if '.jpg' in i:
        i = i.split("/")[-1].replace(".jpg", "")
        read_list.append(int(i))

ee_nn = pd.read_csv(r'C:\Users\Richard\Desktop\ML-AI code bootcamp\Final files\Final image\CollectionImages.csv')

resourceID = ee_nn['Resource ID(s)']
resourceID = list(set(resourceID))
resourceID = [i for i in resourceID if i not in read_list]
print(len(resourceID))

for id in resourceID:
  myID = id
  thumbnail_url = f"https://rs.williams.edu/iiif/image/{myID}/full/150,/0/default.jpg"

  response = requests.get(thumbnail_url)

  if response.status_code == 200:
  # Extract the file name from the URL
    filename = f"{myID}.jpg"

    # Save the image to a file
    with open(filename, "wb") as f:
      f.write(response.content)

    print(f"Image downloaded and saved as '{filename}'.")
  else:
    print("Failed to fetch the image.")

print('finished with everything')