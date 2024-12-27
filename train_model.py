import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Parameters
img_height, img_width = 224, 224
epochs = 20
batch_size = 32
data_dir = r'D:\FinalProject\code\uplaodorcapture\test_data'

# Function to load images and labels
def load_data(data_dir):
    images = []
    labels = []
    class_names = []

    # List all files in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Process image files
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_height, img_width))
            img = img / 255.0  # Normalize the image

            images.append(img)
            # Extract label from filename
            label = filename.split('.')[0]
            if label not in class_names:
                class_names.append(label)
            labels.append(class_names.index(label))  # Convert label to index

    return np.array(images), np.array(labels), class_names

# Load the data
images, labels, class_names = load_data(data_dir)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the data generator on the training data
datagen.fit(X_train)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')  # Number of classes
])

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model using the data generator
model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), 
          validation_data=(X_val, y_val), 
          epochs=epochs)

# Save the model
model.save('constellation_model.keras')  # Save in the recommended format
print(f"Model saved as 'constellation_model.keras'. Classes: {class_names}")




