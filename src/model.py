#the imports I used for this project in order to build a model using MobileNetV2
import os
import ssl
import certifi
# Ensure SSL certificates are set up correctly
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl.create_default_context

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

#Load the data
X = np.load('X.npy')
y = np.load('y.npy')

# Exapand channel to match MobileNetV2 input
X = np.repeat(X, 3, axis=-1)

#Resize images to match MobileNetV2 input size
X_resized = tf.image.resize(X, [224,224]).numpy()

#process for MobileNetV2
X_resized = preprocess_input(X_resized)

#Split the data
X_train, X_val, y_train, y_val = train_test_split(X_resized, y, test_size=0.2, random_state=42)

#Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range = 20,
    width_shift_range= 0.1, 
    height_shift_range = 0.1,
    zoom_range = 0.2,
    horizontal_flip = True,
)

val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

#Build the model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.tainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

#Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

#early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

#Train
model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[early_stop]
)

#evaluation
y_pred = (model.predict(X_val) > 0.5).astype(int)
print(classification_report(y_val, y_pred))

model.save('tumor_model_mbilenet.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('tumor_model_mobilenet.tflite', 'wb') as f:
    f.write(tflite_model)
# Save the model path for later use
MODEL_PATH = 'tumor_model_mobilenet.tflite'
