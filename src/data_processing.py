import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = 224
DATA_DIR = "/Users/ivanaguilarjr/Documents/programs/projects/tumor-detection/brain_tumor_dataset"

categories = ["no", "yes"]
data = []

for catergory in categories:
    path = os.path.join(DATA_DIR, catergory)
    label = categories.index(catergory)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append([img, label])
print(len(data))

import random
random.shuffle(data)

X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X = X / 255.0  # Normalize the data
y = np.array(y)
print(X[0].shape)
print(y[0])

# for i in range(5):
#     print(f"Image {i} shape: {X[i].shape}, Label: {y[i]}")

for i in range (3):
    plt.imshow(X[i].reshape(IMG_SIZE,IMG_SIZE), cmap="gray")
    plt.title(f"Label: {'Tumor' if y[i] == 1 else 'No Tumor'}")
    plt.show()

# Save processed data for use in model.py
np.save('X.npy', X)
np.save('y.npy', y)