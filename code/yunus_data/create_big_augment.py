# This image does pretty much what preprocessing_image_data does. 
# Just that it combines everything into one folder. I wanted to have a clean sheet for this.
# It creates rotated data from yunus_resized in order to then randomly preprocess it with adaptive thresholding, histogram equalization,
# erosion or just leaving it in its original

########## Preparation - Opening and Cutting ##########

#region Opening data
# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import shutil
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import random

train_resized_data_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_train_resized.json'
test_resized_data_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_test_resized.json'
train_img_path = 'yunus_resized/train/'
test_img_path = 'yunus_resized/test/'

# Opening them
with open(train_resized_data_path, 'r') as f:
    df_train_resized_orig = pd.DataFrame(json.load(f))

with open(test_resized_data_path, 'r') as f:
    df_test_resized_orig = pd.DataFrame(json.load(f))


df_train_shuffled = df_train_resized_orig.sample(frac=1, random_state=33333).reset_index(drop=True)
    
# Updated train-validation-test split function - USE WHEN IN NEEED OF VALIDATION DATA
def train_val_split(df):
    train_split = int(0.8 * len(df))

    df_train = df[:train_split]
    df_val = df[train_split:]

    return df_train, df_val

df_train_split, df_val_split = train_val_split(df_train_shuffled)

df_val_split.to_json('/home/ubuntu/MasterThesis/code/yunus_data/df_val_big_aug.json', orient = 'records', indent=4)

df_train_img_list = df_train['img_name'].tolist()
print(len(df_train_img_list)) # 443 observations left

df_big_aug = df_train_split

# Copy images from yunus_resized into yunus_big_augment
# Paths
source_folder_origin_train = "/home/ubuntu/MasterThesis/yunus_resized/train_old/"
source_folder_origin_val = "/home/ubuntu/MasterThesis/yunus_resized/train_old/" # the validation data was previously included here

target_folder = "/home/ubuntu/MasterThesis/yunus_big_augment/"
target_folder_val = '/home/ubuntu/MasterThesis/yunus_resized/validation/'

# Creating big augmentation folder
for img_name in df_train_split['img_name']:
    source_path = os.path.join(source_folder_origin_train, img_name)
    target_path = os.path.join(target_folder, img_name)
    
    # Copy the file if it exists
    if os.path.exists(source_path):
        shutil.copy(source_path, target_path)
    else:
        print(f"Warning: {source_path} does not exist.")

# saving validation pictures in separate folder
for img_name in df_val_split['img_name']:
    source_path = os.path.join(source_folder_origin_val, img_name)
    target_path = os.path.join(target_folder_val, img_name)
    
    # Copy the file if it exists
    if os.path.exists(source_path):
        shutil.copy(source_path, target_path)
    else:
        print(f"Warning: {source_path} does not exist.")
print("Image copying complete.")

#region Check if it worked
# Wie viele Images in Folder? Hat geklappt?
def count_files_in_folder(folder_path):
    length = len(os.listdir(folder_path))
    return length

yunus_big_augment_path = '/home/ubuntu/MasterThesis/yunus_big_augment'
count_original = count_files_in_folder(yunus_big_augment_path)
print(count_original) # 443 images - is right! 

validation_no_augment_path = '/home/ubuntu/MasterThesis/yunus_resized/validation'
count_validation = count_files_in_folder(validation_no_augment_path)
print(count_validation) # 111 images - makes sense!
#endregion

# the training folder in yunus_resized is now renamed to train_old, in order to show that it is not the current training dataset.
print(df_big_aug.columns)
print(len(df_big_aug))


########## Augmenting training dataset ##########
# Rotated images
df_rot_train_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_train_aug_newrotation.json'
# Datensatz beinhaltet alten Trainings + Validierungsdatensatz

with open(df_rot_train_path, 'r') as f:
    df_rot_train = pd.DataFrame(json.load(f))

def filter_and_append_rotated_data(df_big_aug, df_other):
    """
    Filters rows from df_other where img_name matches the original img_name or its rotated variants
    based on df_big_aug, and appends them to df_big_aug.

    Parameters:
        df_big_aug (pd.DataFrame): Original dataset with image metadata.
        df_other (pd.DataFrame): Dataset with additional rotated image metadata.

    Returns:
        pd.DataFrame: Updated df_big_aug with relevant rows from df_other appended.
    """
    import re

    # Extract original image names without ".jpg"
    original_img_names = df_big_aug['img_name'].str.replace('.jpg', '', regex=False).unique()
    
    # Prepare explicit list of allowed filenames
    allowed_filenames = set()
    for name in original_img_names:
        allowed_filenames.add(f"{name}_+5.jpg")
        allowed_filenames.add(f"{name}_-5.jpg")

    # Filter df_other for rows where img_name is in the allowed list
    filtered_df = df_other[df_other['img_name'].isin(allowed_filenames)]

    # Append filtered rows to df_big_aug
    updated_df = pd.concat([df_big_aug, filtered_df], ignore_index=True)
    
    return updated_df

df_orig_rot = filter_and_append_rotated_data(df_big_aug, df_rot_train)

print(df_orig_rot['img_name'].head(30))
print(len(df_orig_rot))
df_train_with_rotation = df_orig_rot

# 
target_folder = "/home/ubuntu/MasterThesis/yunus_big_augment/"
df_rot_train_images = '/home/ubuntu/MasterThesis/yunus_aug_rotation/train'
df_rot_val_images = '/home/ubuntu/MasterThesis/yunus_aug_rotation/validation'

for img_name in df_orig_rot['img_name']:
    source_path = os.path.join(df_rot_val_images, img_name)
    target_path = os.path.join(target_folder, img_name)
    
    if os.path.exists(source_path):
        shutil.copy(source_path, target_path)
    else:
        print(f"Warning: {source_path} does not exist.")

#region Check if it worked
# Wie viele Images in Folder? Hat geklappt?
def count_files_in_folder(folder_path):
    length = len(os.listdir(folder_path))
    return length

big_augment_rotation_path = '/home/ubuntu/MasterThesis/yunus_big_augment'
count_after_rotation = count_files_in_folder(big_augment_rotation_path)
print(count_after_rotation) # 1329 - ALLE SIIND DA !!
print(len(df_train_with_rotation))
print(df_train_with_rotation.head(20))

df_train_with_rotation.to_json('/home/ubuntu/MasterThesis/code/yunus_data/df_train_big_aug.json', orient='records', indent=4)

#endregion


def random_image_modification(input_folder, output_folder):
    """
    Randomly modifies images from the input folder and saves the results in the output folder.

    Modifications (25% chance each):
        - Keep the image as it is (in color)
        - grayscale + adaptive thresholding
        - grayscale + histogram equalization
        - grayscale + erosion

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where processed images will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)

        image = cv2.imread(filepath)

        # Skip if failed to load
        if image is None or image.size == 0:
            print(f"Warning: Failed to load image or image is empty: {filepath}")
            continue

        # Randomly select an operation
        operation = random.choice(['color', 'adaptive_thresholding', 'histogram_equalization', 'erosion'])

        try:
            if operation == 'color':
                # Keep the image in color
                processed_image = image

            elif operation == 'adaptive_thresholding':
                # Convert to grayscale and apply adaptive thresholding
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                processed_image = cv2.adaptiveThreshold(
                    gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )

            elif operation == 'histogram_equalization':
                # Convert to grayscale and apply histogram equalization
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                processed_image = cv2.equalizeHist(gray_image)

            elif operation == 'erosion':
                # Convert to grayscale and apply erosion
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                kernel = np.ones((3, 3), np.uint8)
                processed_image = cv2.erode(gray_image, kernel, iterations=1)

            # Saving 
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, processed_image)
            print(f"Processed ({operation}) and saved: {output_path}")

        except Exception as e:
            print(f"Error processing {filepath}: {e}")


input_folder = "/home/ubuntu/MasterThesis/yunus_big_augment"
output_folder = "/home/ubuntu/MasterThesis/yunus_random_augment"
random_image_modification(input_folder, output_folder)


#region Checken, dass alles zusammenpasst
import os
import pandas as pd

def check_image_text_alignment(root_dir, data_path, img_column='img_name'):
    """
    Check if the images in a directory match the text data in a JSON file.

    Args:
        root_dir (str): Path to the folder containing images.
        data_path (str): Path to the JSON file containing text data with 'img_name'.
        img_column (str): Column name in the JSON file containing image names.

    Returns:
        None
    """
    data = pd.read_json(data_path)
    image_files = set(os.listdir(root_dir))

    # list of names
    dataset_images = set(data[img_column])

    # Check for mismatches
    missing_in_folder = dataset_images - image_files  # Images in the dataset but not in the folder
    missing_in_dataset = image_files - dataset_images  # Images in the folder but not in the dataset

    # Print results
    print(f"Results for {root_dir} and {data_path}:")
    if missing_in_folder:
        print(f"Images in the dataset but missing in folder: {len(missing_in_folder)}")
        print(missing_in_folder)
    else:
        print("All dataset images are present in the folder.")

    if missing_in_dataset:
        print(f"Images in the folder but missing in the dataset: {len(missing_in_dataset)}")
        print(missing_in_dataset)
    else:
        print("All folder images are present in the dataset.")
    print("\n")

# Paths
root_dir_train = "/home/ubuntu/MasterThesis/yunus_random_augment"
root_dir_test = "/home/ubuntu/MasterThesis/yunus_resized/test"
root_dir_val = '/home/ubuntu/MasterThesis/yunus_resized/validation'
train_data_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_train_big_aug.json'
test_data_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_test_resized.json'
val_data_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_val_big_aug.json'

# Check alignment for training, test, and validation data
check_image_text_alignment(root_dir_train, train_data_path)
check_image_text_alignment(root_dir_test, test_data_path)
check_image_text_alignment(root_dir_val, val_data_path)

# Alles passt zusammen. Juhu!
#endregion

