import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the saved model
model = load_model('C:/Users/Admin/Desktop/manhole/model.h5')

path = "C:/Users/Admin/Desktop/manhole/test"
files = os.listdir(path)

# Function to preprocess the image
def preprocess_image(path):
    img = image.load_img(path, target_size=(224, 224))  # Assuming 224x224 input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

for i in files:
    if not i.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Skipping non-image file: {i}")
        continue  # Skip non-image files

    pth = os.path.join(path, i)
    X = cv2.imread(pth, cv2.IMREAD_COLOR)

    if X is None or X.size == 0:
        print(f"Unable to read or empty image: {pth}")
        continue  # Skip to the next image if current one is problematic

    X = cv2.resize(X, (256, 256))

    plt.figure()
    plt.imshow(X[:, :, ::-1])
    plt.show()

    X = np.array(X)
    X = np.expand_dims(X, axis=0)

    y_pred = np.round(model.predict(X))
    if np.array_equal(y_pred[0], [1, 0]):
        print("Manhole Close")
    else:
        print("Manhole Open")

        
