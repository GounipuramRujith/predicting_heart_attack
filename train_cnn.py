# train_cnn.py
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import zipfile
import shutil # Import shutil for file operations like copy

# --- Configuration for Data Paths ---
# IMPORTANT: Use the ABSOLUTE paths where your ZIP files are located.
MI_ZIP_PATH = '/Users/rujith/Downloads/gwbz3fsgp8-2/ECG Images of Myocardial Infarction Patients (240x12=2880).zip'
NORMAL_ZIP_PATH = '/Users/rujith/Downloads/gwbz3fsgp8-2/Normal Person ECG Images (284x12=3408).zip'

# Define the target directories where the images will be unzipped
# These are the names of the folders that will be created in your script's directory
MI_BASE_UNZIPPED_DIR = 'ECG Images of Myocardial Infarction Patients (240x12=2880)'
NORMAL_BASE_UNZIPPED_DIR = 'Normal Person ECG Images (284x12=3408)'

# Define the new flat directories where all images will be copied
MI_FLAT_DIR = 'mi_flat_images'
NORMAL_FLAT_DIR = 'normal_flat_images'

# --- Helper function to check if a file is an image ---
def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))

# --- Function to flatten image directories ---
def flatten_image_directory(source_dir, target_flat_dir):
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' not found for flattening.")
        return

    os.makedirs(target_flat_dir, exist_ok=True) # Create the target flat directory if it doesn't exist
    print(f"Flattening images from '{source_dir}' to '{target_flat_dir}'...")
    
    copied_count = 0
    for root, _, files in os.walk(source_dir):
        for file in files:
            if is_image_file(file):
                source_file_path = os.path.join(root, file)
                target_file_path = os.path.join(target_flat_dir, file)
                # Avoid overwriting if file names are not unique across subfolders
                # You might want to add a unique identifier if filenames repeat
                if not os.path.exists(target_file_path):
                    shutil.copy(source_file_path, target_file_path)
                    copied_count += 1
                else:
                    # If a file with the same name exists, append a counter
                    name, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(os.path.join(target_flat_dir, f"{name}_{counter}{ext}")):
                        counter += 1
                    shutil.copy(source_file_path, os.path.join(target_flat_dir, f"{name}_{counter}{ext}"))
                    copied_count += 1

    print(f"Finished flattening: Copied {copied_count} image files to '{target_flat_dir}'.")


# Function to load ECG images from a given folder
def load_img_from_folder(path, label, size=(224, 224)):
  images = []
  labels = []
  
  if not os.path.isdir(path):
      print(f"Error: Image directory '{path}' not found. Please ensure it's unzipped and the path is correct.")
      return [], []
  
  files_in_dir = os.listdir(path)
  if not files_in_dir:
      print(f"Error: Image directory '{path}' is empty. No files found.")
      return [], []

  print(f"Attempting to load images from: {path}")
  loaded_count = 0
  skipped_count = 0

  for fname in files_in_dir:
    # Skip hidden files like .DS_Store or __MACOSX directories
    if fname.startswith('.') or fname == '__MACOSX':
        skipped_count += 1
        continue

    imag_path = os.path.join(path, fname)
    
    # If it's a directory, skip it (shouldn't happen in flat directory)
    if os.path.isdir(imag_path):
        skipped_count += 1
        continue
    
    # Attempt to read the image
    img = cv2.imread(imag_path, cv2.IMREAD_GRAYSCALE)
    
    if img is not None:
      img = cv2.resize(img, size)
      img = img / 255.0
      images.append(img)
      labels.append(label)
      loaded_count += 1
    else:
      # Print a message if an image fails to load
      print(f"Warning: Could not read file as image: {imag_path}")
      skipped_count += 1
      
  print(f"Finished loading from {path}: Loaded {loaded_count} images, Skipped {skipped_count} non-image/hidden files.")
  return images, labels

# --- Unzipping Logic ---
def unzip_data_if_needed(zip_path, target_dir):
    if not os.path.exists(target_dir):
        print(f"Unzipping '{zip_path}' to '{target_dir}'...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            print("Unzipping complete.")
        except FileNotFoundError:
            print(f"Error: ZIP file '{zip_path}' not found. Please ensure it's in the correct directory.")
            exit()
        except Exception as e:
            print(f"An error occurred during unzipping '{zip_path}': {e}")
            exit()
    else:
        print(f"Directory '{target_dir}' already exists. Skipping unzipping.")

# --- Perform Unzipping ---
unzip_data_if_needed(MI_ZIP_PATH, MI_BASE_UNZIPPED_DIR)
unzip_data_if_needed(NORMAL_ZIP_PATH, NORMAL_BASE_UNZIPPED_DIR)

# --- Flatten the unzipped directories ---
flatten_image_directory(MI_BASE_UNZIPPED_DIR, MI_FLAT_DIR)
flatten_image_directory(NORMAL_BASE_UNZIPPED_DIR, NORMAL_FLAT_DIR)

# --- Load MI and Normal Images from the NEW FLAT directories ---
mi_imgs, mi_labels = load_img_from_folder(MI_FLAT_DIR, label=1)
normal_imgs, normal_labels = load_img_from_folder(NORMAL_FLAT_DIR, label=0)

# Check if images were loaded successfully
if not mi_imgs or not normal_imgs:
    print("Exiting: Could not load images from flat directories. Please check paths and contents.")
    if not mi_imgs:
        print(f"  - No MI images were loaded from '{MI_FLAT_DIR}'.")
    if not normal_imgs:
        print(f"  - No Normal images were loaded from '{NORMAL_FLAT_DIR}'.")
    exit()

# Combine both sets of images and labels
x = np.array(mi_imgs + normal_imgs)
y = np.array(mi_labels + normal_labels)

 # Reshape to add channel dimension: (samples, height, width, channels)
x = x.reshape(-1, 224, 224, 1)

# --- Model Training ---
from sklearn.model_selection import train_test_split

# Split data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=42)

print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

import tensorflow as tf
from tensorflow.keras import layers, models

# Build a Convolutional Neural Network
model = models.Sequential()

# 1st Convolutional Layer: 32 filters, 3x3 kernel, ReLU activation
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(layers.MaxPooling2D(2, 2)) # Downsample by 2x

# 2nd Convolutional Layer: 64 filters
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2)) # Downsample again

# Flatten the output from conv layers to feed into Dense layer
model.add(layers.Flatten())


# Fully connected layer with 128 neurons and ReLU
model.add(layers.Dense(128, activation='relu'))

# Output layer: 1 neuron with sigmoid (binary classification)
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Train the Model
print("\nTraining CNN Model...")
training = model.fit(x_train, y_train, epochs=10, batch_size=16,
                     validation_data = (x_test, y_test))

# Save the trained model
model.save('cnn_model.h5')
print("CNN model saved as cnn_model.h5")

