import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt

calibrated_image_directory = "/Users/giovannilopez/Downloads/2024-08-15_Cultivos/calibrated/"


# Build the CNN Model for Object Detection
# Define a simple CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4)  # 4 outputs for bounding box (x, y, width, height)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model


model = build_model()
model.summary()


# Prepare the Data Generators
# Define data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,                # Normalizes pixel values
    rotation_range=40,             # Randomly rotates images
    width_shift_range=0.2,         # Randomly shifts images horizontally
    height_shift_range=0.2,        # Randomly shifts images vertically
    shear_range=0.2,               # Randomly applies shearing transformations
    zoom_range=0.2,                # Randomly zooms into images
    horizontal_flip=True,          # Randomly flips images horizontally
    fill_mode='nearest'            # Fills in missing pixels after transformations
)

train_generator = train_datagen.flow_from_directory(
    calibrated_image_directory,
    target_size=(128, 128),
    batch_size=32,
    class_mode='sparse'
)


val_datagen = ImageDataGenerator(rescale=1./255)  # Only normalization, no augmentation

val_generator = val_datagen.flow_from_directory(
    calibrated_image_directory,
    target_size=(128, 128),
    batch_size=32,
    class_mode='sparse'
)

# Note: Custom data generator might be needed to yield (image, bounding_box) pairs


# Train the model
model.fit(train_generator, epochs=10, validation_data=val_generator)


def predict_and_visualize(image_path, model):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (128, 128))
    img_array = np.expand_dims(img_resized, axis=0) / 255.0

    # Predict bounding box
    pred = model.predict(img_array)[0]  # [x, y, width, height]

    # Scale predictions back to original image size
    h, w, _ = img.shape
    x, y, width, height = pred * [w, h, w, h]

    # Draw bounding box
    start_point = (int(x), int(y))
    end_point = (int(x + width), int(y + height))
    color = (0, 255, 0)  # Green
    thickness = 2
    img_box = cv2.rectangle(img, start_point, end_point, color, thickness)

    # Show the image
    plt.imshow(cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# Example usage
predict_and_visualize('/Users/giovannilopez/Downloads/2024-08-15_Cultivos/calibrated/flower_DSC_4574.jpg', model)
