#import necessary libaries
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import os

#Path to where the images are stored
path = 'C:\\Users\\Richard\\Desktop\\ML-AI code bootcamp\\Final files\\Final image'

#setting the directory to the path
dir_list = os.listdir(path)

#Empty list that will be used to store what images have already been downloaded
read_list = []
#Finding which images have already been downloaded and adding it to the list
for i in dir_list:
    if '.jpg' in i:
        i = i.split("/")[-1].replace(".jpg", "")
        read_list.append(int(i))

#Reading the provided metadata csv spreadsheet
ee_nn = pd.read_csv(r'C:\Users\Richard\Desktop\ML-AI code bootcamp\Final files\Final image\CollectionImages.csv')

#Setting what images to download by
resourceID = ee_nn['Resource ID(s)_x']
resourceID = list(set(resourceID))
#removing already downloaded images from the to-download list
resourceID = [i for i in resourceID if i not in read_list]
print(len(resourceID))

#Loacating and downloading every image from the spreadsheet that has not yet been downloaded
for id in resourceID:
#image to download
  myID = id
#website to download from
  thumbnail_url = f"https://rs.williams.edu/iiif/image/{myID}/full/150,/0/default.jpg"
#getting the image from the website
  response = requests.get(thumbnail_url)
#deciding to download the image as a jpg
  if response.status_code == 200:
  # Extract the file name from the URL
    filename = f"{myID}.jpg"

    # Save the image to a new file
    with open(filename, "wb") as f:
      f.write(response.content)
#Printing the image was downloaded
    print(f"Image downloaded and saved as '{filename}'.")
  else:
#print if image isn't downloaded
    print("Failed to fetch the image.")

#Prints after all images in the dataset are finished downloading
print('finished all labeled image')
