import os
import numpy as np
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

categories = ["ants", "bees", "beetle", "catterpillar", "earthworms", "earwig", "grasshopper", "moth", "slug", "snail", "wasp", "weevil"]
img_size = 80

def load_images(data_dir, data_list, labels_list):
    for idx, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        for img_name in os.listdir(category_path):
            try:
                img_path = os.path.join(category_path, img_name)
                img = Image.open(img_path).convert("L").resize((img_size, img_size))
                data_list.append(np.array(img))
                labels_list.append(idx)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
def train_pest_model(train_data, train_labels):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(categories), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    datagen.fit(train_data)
    model.fit(datagen.flow(train_data, train_labels, batch_size=32), epochs=20)
    model.save("pest_classification_model_v2.h5")
    return model
if __name__ == "__main__":
    train_data_dir = "D:/Intern/train"
    train_data, train_labels = [], []
    load_images(train_data_dir, train_data, train_labels)
    train_data = np.array(train_data).reshape(-1, img_size, img_size, 1) / 255.0
    train_labels = np.array(train_labels)
    pest_model = train_pest_model(train_data, train_labels)
