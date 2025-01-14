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

train_data_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_train.json'
val_data_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_val.json'
train_img_path = 'yunus_photos/train2017/'
val_img_path = 'yunus_photos/val2017/'
train_resized_data_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_train_resized.json'
val_resized_data_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_val_resized.json'

# Opening them
with open(train_data_path, 'r') as f:
    train_img_data = pd.DataFrame(json.load(f))

with open(val_data_path, 'r') as f:
    val_img_data = pd.DataFrame(json.load(f))

with open(train_resized_data_path, 'r') as f:
    df_train_resized = pd.DataFrame(json.load(f))

with open(val_resized_data_path, 'r') as f:
    df_val_resized = pd.DataFrame(json.load(f))

# Concatenation of Train & Validation Data
img_data_concat = pd.concat([train_img_data, val_img_data], ignore_index=True)
print(img_data_concat.head())
print(len(img_data_concat))
print(img_data_concat['bbox'].head())

# Bounding Boxes per Image based on concatenated dataset and ORIGINAL bounding boxes
bboxes_per_img = img_data_concat.set_index("img_name")["bbox"].to_dict()
#for key, value in list(bboxes_per_img.items())[:2]:
#    print(f"{key}: {value}")

# Bounding Boxes with INDICES in order to test order in dataset vs image
bboxes_with_indices = {
    img_name: {idx: bbox for idx, bbox in enumerate(bboxes, start=1)}
    for img_name, bboxes in bboxes_per_img.items()
}
#for key, value in list(bboxes_with_indices.items())[:2]:
#    print(f"{key}: {value}")

# Dictionary of ABZ notations per image - in order to be linked to Bounding Box Data
abz_dict = img_data_concat.set_index("img_name")["abz"].to_dict()  


# Wie viele Images in Folder
def count_files_in_folder(folder_path):
    length = len(os.listdir(folder_path))
    return length

train_img_resized_path = "/home/ubuntu/MasterThesis/train_set"
val_img_resized_path = "/home/ubuntu/MasterThesis/yunus_resized/val"
count_train_img_resized = count_files_in_folder(train_img_resized_path)
count_val_img_resized = count_files_in_folder(val_img_resized_path)
print(f"Number of files in the training folder: {count_train_img_resized}, \nNumber of files in the validation folder: {count_val_img_resized}")
print(count_train_img_resized)
#endregion


#region ALREADY EXECUTED Cutting Images
def process_images_in_folder_with_lines(folder_path):
    """
    Process all images in a folder and draw horizontal lines every 500 pixels to indicate potential cutting points.

    Args:
        folder_path (str): Path to the folder containing images.
    """
    # for all files in folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Load 
        image = cv2.imread(file_path)
        height, width, _ = image.shape

        # Draw horizontal lines every 500 pixels
        display_image = image.copy()
        line_spacing = 500

        for y in range(0, height, line_spacing):
            cv2.line(display_image, (0, y), (width, y), (0, 255, 0), 2)  # Green lines
            cv2.putText(display_image, f"y={y}", (10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)  # y-coordinate of lines

        # Display imges with lines
        plt.figure(figsize=(10, 10))
        plt.title(f"Cutting Points for {file_name}")
        plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

def cut_image_at_line(image_path, line_y, output_folder):
    """
    Cut an image horizontally at a specified line and save the resulting parts.

    Args:
        image_path (str): Path to the image to be cut.
        line_y (int): The y-coordinate of the horizontal line to cut at.
        output_folder (str): Folder to save the resulting parts.
    """
    # Load 
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    height, width, _ = image.shape

    # create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Split img into two parts at the line_y
    top_part = image[:line_y, :]
    bottom_part = image[line_y:, :]

    # Save both parts
    file_name = os.path.basename(image_path)
    top_path = os.path.join(output_folder, f"{file_name}_top.jpg")
    bottom_path = os.path.join(output_folder, f"{file_name}_bottom.jpg")

    cv2.imwrite(top_path, top_part)
    cv2.imwrite(bottom_path, bottom_part)

    print(f"Saved top part to {top_path}")
    print(f"Saved bottom part to {bottom_path}")


folder_path = "test_photos/"
output_folder = "extracted_tablets"
process_images_in_folder_with_lines(folder_path)

#region Cuttings
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.19888.jpg',592,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.13355.jpg',1072,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.22981.jpg',425,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.19118.jpg',798,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/F.76.jpg',1736,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/BM.41556.jpg',2745,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.13731.jpg',1187,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.14854.jpg',1216,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/BM.39788.jpg',1106,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.12984.jpg',832,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.21803.jpg',749,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.2638.jpg',1176,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.18875.jpg',693,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.19000.jpg',896,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.19001.jpg',791,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/BM.32376.jpg',2916,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.19010.jpg',791,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.20493.jpg',1160,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.138878.jpg',1200,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.20256.jpg',808,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.14481.jpg',896,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.12999.jpg',1112,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.10208.jpg',2000,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.15827.jpg',1128,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.19006.jpg',791,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.16446.jpg',826,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/SP-III.917.jpg',1568,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.19403.jpg',826,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/1891,0509.236.jpg',822,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.17677.jpg',1200,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.12687.jpg',1080,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.10084.jpg',1421,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.4782.jpg',1520,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.9499.jpg',1940,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/BM.40431.jpg',1528,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.17666.jpg',889,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.19105.jpg',600,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/BM.114741.jpg',3100,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/VAT13604Rs-0',5000,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/VAT10657-0.jpg',3000,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.2025.jpg',3000,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/VAT11100Rs-0.jpg',3250,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/VAT10601-0.jpg',3500,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/K.13887.jpg',1225,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/ND.6201.jpg',4000,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/BM.33550.jpg',5000,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/BM.40757.jpg',5000,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/1880,0719.194.jpg',3500,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/photos_to_be_cut/BM.41665.jpg',3100,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/letzte_fotos/BM.41521.jpg',9400,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/letzte_fotos/K.18767.jpg',1000,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/letzte_fotos/BM.45156.jpg',2600,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/letzte_fotos/VAT13604Vs-0.jpg',4850,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/letzte_fotos/BM.34225.jpg',2700,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/letzte_fotos/VAT13604Rs-0.jpg',5000,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/letzte_fotos/BM.35224.jpg',4500,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/letzte_fotos/BM.41176.jpg',2700,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/letzte_fotos/BM.41665.jpg',3140,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/test_photos/BM.34225.jpg',2000,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/test_photos/BM.40406.jpg',2500,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/test_photos/F.76.jpg',950,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/test_photos/BM.41556.jpg',2130,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/test_photos/BM.32376.jpg',1000,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/test_photos/BM.35224.jpg',1490,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/test_photos/BM.41176.jpg',980,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/test_photos/BM.33550.jpg',1750,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/test_photos/BM.40757.jpg',4200,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/test_photos/1880,0719.194.jpg',1150,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')
#cut_image_at_line('/Users/irina/PythonProjects/MasterThesis/test_photos/BM.41665.jpg',900,'/Users/irina/PythonProjects/MasterThesis/extracted_tablets')#


#endregion
#endregion

#region Cutting Image Problem: Need cutted images already ordered in training and validation dataset as later
import os
import shutil
import pandas as pd

# Paths
cutted_images_folder = "/home/ubuntu/MasterThesis/cutted_images"
train_folder = "/home/ubuntu/MasterThesis/cutted_images/train"
val_folder = "/home/ubuntu/MasterThesis/cutted_images/val"

# Ensure train and val folders exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Extract image names from the datasets 
train_img_names = set(df_train_resized['img_name'].values)
val_img_names = set(df_val_resized['img_name'].values)

# Copy images to the folders
for img_name in os.listdir(cutted_images_folder):
    if img_name in train_img_names:
        shutil.copy(os.path.join(cutted_images_folder, img_name), os.path.join(train_folder, img_name))
    elif img_name in val_img_names:
        shutil.copy(os.path.join(cutted_images_folder, img_name), os.path.join(val_folder, img_name))

train_file_folder = "/home/ubuntu/MasterThesis/cutted_images/train"  
val_file_folder = "/home/ubuntu/MasterThesis/cutted_images/val" 

# Count files in the folder (no filtering for file type here)
train_count = len(os.listdir(image_folder))
val_count = len(os.listdir(val_file_folder))

train_count
val_count


#endregion

########## Overview over Image Data and Bounding Box Situation ##########

#region Display9ing images with bounding boxes, with ABZ notation, and save them
def process_images_in_folder(folder_path, bounding_boxes=None):
    """
    Display images with their bounding boxes and index numbers.

    Args:
        folder_path (str): Path to the folder containing images.
        bounding_boxes (dict): Existing bounding boxes for images, where keys are image file names and values are lists of bounding boxes.
    """
    # for all files in folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Load the image
        image = cv2.imread(file_path)

        # Draw bounding boxes
        display_image = image.copy()
        if bounding_boxes and file_name in bounding_boxes:
            for idx, bbox in enumerate(bounding_boxes[file_name], start=1):
                x, y, w, h = bbox
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Add index to top-left corner of bbox
                cv2.putText(display_image, str(idx), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        plt.figure(figsize=(10, 10))
        plt.title(f"Bounding Boxes for {file_name}")
        plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

def process_and_save_images(folder_path, bounding_boxes, output_folder):
    """
    Display images with their bounding boxes and save the results.

    Args:
        folder_path (str): Path to the folder containing images.
        bounding_boxes (dict): Existing bounding boxes for images, where keys are image file names and values are lists of bounding boxes.
        output_folder (str): Folder to save images with bounding boxes drawn.
    """

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image = cv2.imread(file_path)

        if image is None:
            print(f"Failed to load image: {file_path}")
            continue

        display_image = image.copy()
        if bounding_boxes and file_name in bounding_boxes:
            for idx, bbox in enumerate(bounding_boxes[file_name], start=1):
                x, y, w, h = bbox
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (255, 0, 0), 2) # bboxes
                cv2.putText(display_image, str(idx), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2) # index

        # Added: Saving image
        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, display_image)
        print(f"Saved: {output_path}")

        # also looking at them
        plt.figure(figsize=(10, 10))
        plt.title(f"Bounding Boxes for {file_name}")
        plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

def process_and_save_images_with_abz(folder_path, bounding_boxes, abz_data, output_folder):
    """
    Display images with their bounding boxes, index numbers, and abz notation, and save the results.

    Args:
        folder_path (str): Path to the folder containing images.
        bounding_boxes (dict): Existing bounding boxes for images, where keys are image file names and values are lists of bounding boxes.
        abz_data (dict): Dictionary with image names as keys and lists of abz notations as values.
        output_folder (str): Folder to save images with bounding boxes and abz notation drawn.
    """
    os.makedirs(output_folder, exist_ok=True) 

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image = cv2.imread(file_path)

        if image is None:
            print(f"Failed to load image: {file_path}")
            continue

        display_image = image.copy()
        if bounding_boxes and file_name in bounding_boxes:
            for idx, (bbox, abz) in enumerate(zip(bounding_boxes[file_name], abz_data.get(file_name, [])), start=1):
                x, y, w, h = bbox
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Add index and abz notation to the top-left corner of the bounding box
                text = f"{idx}: {abz}"
                cv2.putText(display_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 0, 0), 2)

        # Save
        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, display_image)
        print(f"Saved: {output_path}")

        #  display
        plt.figure(figsize=(10, 10))
        plt.title(f"Bounding Boxes for {file_name}")
        plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

def save_images_with_abz(folder_path, bounding_boxes, abz_data, output_folder):
    """
    Save images with their bounding boxes and corresponding abz annotations to an output folder.

    Args:
        folder_path (str): Path to the folder containing images.
        bounding_boxes (dict): Bounding boxes for images, where keys are image file names and values are lists of bounding boxes.
        abz_data (dict): Dictionary with image names as keys and lists of abz annotations as values.
        output_folder (str): Path to the folder where processed images will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image = cv2.imread(file_path)

        if image is None:
            print(f"Failed to load image: {file_path}")
            continue

        display_image = image.copy()

        # bbox and abz
        if bounding_boxes and file_name in bounding_boxes:
            for idx, (bbox, abz) in enumerate(zip(bounding_boxes[file_name], abz_data.get(file_name, [])), start=1):
                x, y, w, h = bbox
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Add abz annotation to the top-left corner of the bounding box
                cv2.putText(display_image, str(abz), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 0, 0), 2)

        # saving
        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, display_image)
        print(f"Saved: {output_path}")
##### Use functions to display images, save a fraction of test images for thesis in a folder #####

process_images_in_folder('thesis_photos/original', bboxes_per_img)
process_and_save_images('thesis_photos/original', bboxes_per_img, 'thesis_photos/bboxes_original/index' )
process_and_save_images_with_abz('thesis_photos/original', bboxes_per_img, abz_dict, 'thesis_photos/bboxes_original')

# Bounding Boxes are a mess for some images! 
# It is visible that images have different sizes 

# In the beginning also did this with train and test dataset, but noted that the bounding boxes are not in the right order
#process_images_in_folder(folder_path='/Users/irina/PythonProjects/MasterThesis/yunus_photos/val2017', bounding_boxes=bboxes_per_img)

#endregion

#region Summary Statistics functions

# Per Image
def summarize_bounding_boxes(bounding_boxes):
    """
    Calculate summary statistics for bounding boxes for each image.

    Args:
        bounding_boxes (dict): Dictionary where keys are image names and values are lists of its bounding boxes [x, y, w, h].

    Returns:
        pd.DataFrame: Summary statistics per image.
    """
    summary_stats = []

    for img_name, bboxes in bounding_boxes.items():
        widths = [bbox[2] for bbox in bboxes]
        heights = [bbox[3] for bbox in bboxes]
        areas = [w * h for w, h in zip(widths, heights)]

        summary_stats.append({
            "image": img_name,
            "num_bboxes": len(bboxes),
            "median_width": np.median(widths),
            "median_height": np.median(heights),
            "median_area": np.median(areas),
            "mean_width": np.mean(widths),
            "mean_height": np.mean(heights),
            "mean_area": np.mean(areas),
            "std_width": np.std(widths),
            "std_height": np.std(heights),
            "std_area": np.std(areas),
            "min_width": np.min(widths),
            "max_width": np.max(widths),
            "min_height": np.min(heights),
            "max_height": np.max(heights),
            "heights": heights

        })

    return pd.DataFrame(summary_stats)

# For ALL Images
def summarize_all_bounding_boxes(bounding_boxes):
    """
    Calculate summary statistics for all bounding boxes ACROSS ALL IMAGES.

    Args:
        bounding_boxes (dict): Dictionary where keys are image names and values are lists of bounding boxes [x, y, width, height].

    Returns:
        dict: Summary statistics for all bounding boxes.
    """
    all_widths = []
    all_heights = []

    # Iterate over all bounding boxes
    for img_name, bboxes in bounding_boxes.items():
        all_widths.extend([bbox[2] for bbox in bboxes])  # Extract widths
        all_heights.extend([bbox[3] for bbox in bboxes])  # Extract heights

    if not all_widths or not all_heights:
        return {
            "total_bboxes": 0,
            "median_width": None,
            "median_height": None,
            "mean_width": None,
            "mean_height": None,
            "std_width": None,
            "std_height": None,
            "min_width": None,
            "max_width": None,
            "min_height": None,
            "max_height": None,
        }

    # Calculate statistics
    stats = {
        "total_bboxes": len(all_widths),
        "median_width": np.median(all_widths),
        "median_height": np.median(all_heights),
        "mean_width": np.mean(all_widths),
        "mean_height": np.mean(all_heights),
        "std_width": np.std(all_widths),
        "std_height": np.std(all_heights),
        "min_width": np.min(all_widths),
        "max_width": np.max(all_widths),
        "min_height": np.min(all_heights),
        "max_height": np.max(all_heights),
    }
    return stats

#endregion

#region Getting Summary Statistics on ORIGINAL Image Bounding Boxes
pd.set_option('display.max_rows', None)       
pd.set_option('display.max_columns', None)    
pd.set_option('display.width', 1000)          
pd.set_option('display.max_colwidth', None) 

stats_per_image = summarize_bounding_boxes(bboxes_per_img)
print(stats_per_image.head(10))

# bbox height in these 10 images varies between 28 and 357
# bbox width varies between 29 and 506

# means and medians vary significantly as well. 
# std in height are smaller than in width. 


stats_all = summarize_all_bounding_boxes(bboxes_per_img)
print(stats_all)
# Summary for all images before resizing
# {'total_bboxes': 54999, 'median_width': 137.0, 'median_height': 116.0, 'mean_width': 166.40760741104384, 'mean_height': 141.4401898216331, 
# 'std_width': 106.05473748489094, 'std_height': 80.46100817146315, 'min_width': 13, 'max_width': 2445, 'min_height': 17, 'max_height': 2118}

# Median height is 116, with min value of 17 and max 2118. Std of height is 80.
# Median width is 137, with min value of 13 and max 2445. Std of width is 106.

# Get the first key and value
first_key = next(iter(bboxes_per_img))
first_value = bboxes_per_img[first_key]

print("First key:", first_key)
print("First value:", first_value)
#endregion



########## Resizing and rotating Images ##########

#region Resize images to having similar size based on having bounding boxes similar in height

def resize_image_and_boxes(image, bboxes, target_bbox_height, max_image_size=(1024, 1024), final_size=(1024,1024)):
    """
    Resize an image based on bounding box height and apply additional resizing and padding.

    Args:
        image (np.array): Input image.
        bboxes (list): List of bounding boxes [x, y, w, h].
        target_bbox_height (int): Target median bounding box height.
        max_image_size (tuple): Maximum allowable image size (height, width).
        final_size (tuple): Final quadratic size for padding.

    Returns:
        np.array: Final resized and padded image.
        list: Adjusted bounding boxes.
    """
    original_h, original_w = image.shape[:2]
    median_bbox_height = np.median([bbox[3] for bbox in bboxes]) if bboxes else 0
    scaling_factor = target_bbox_height / median_bbox_height if median_bbox_height > 0 else 1.0

    # Rescale image to match target bounding box height
    new_h, new_w = int(original_h * scaling_factor), int(original_w * scaling_factor)
    resized_image = cv2.resize(image, (new_w, new_h))
    scaled_bboxes = [[int(x * scaling_factor), int(y * scaling_factor), int(w * scaling_factor), int(h * scaling_factor)] for x, y, w, h in bboxes]

    # Ensure image does not exceed max_image_size - if it does, resize
    max_h, max_w = max_image_size
    if new_h > max_h or new_w > max_w:
        scaling_factor = min(max_w / new_w, max_h / new_h)
        new_h, new_w = int(new_h * scaling_factor), int(new_w * scaling_factor)
        resized_image = cv2.resize(resized_image, (new_w, new_h))
        scaled_bboxes = [[int(x * scaling_factor), int(y * scaling_factor), int(w * scaling_factor), int(h * scaling_factor)] for x, y, w, h in scaled_bboxes]

    # Check final size before padding
    final_h, final_w = final_size
    if new_h > final_h or new_w > final_w:
        # Rescale again if necessary to fit within final size
        scaling_factor = min(final_w / new_w, final_h / new_h)
        new_h, new_w = int(new_h * scaling_factor), int(new_w * scaling_factor)
        resized_image = cv2.resize(resized_image, (new_w, new_h))
        scaled_bboxes = [[int(x * scaling_factor), int(y * scaling_factor), int(w * scaling_factor), int(h * scaling_factor)] for x, y, w, h in scaled_bboxes]

    # Pad image to final_size
    pad_top = max((final_h - new_h) // 2, 0)
    pad_bottom = max(final_h - new_h - pad_top, 0)
    pad_left = max((final_w - new_w) // 2, 0)
    pad_right = max(final_w - new_w - pad_left, 0)

    padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    padded_bboxes = [[x + pad_left, y + pad_top, w, h] for x, y, w, h in scaled_bboxes]

    return padded_image, padded_bboxes


input_folder = "/home/ubuntu/MasterThesis/cutted_images"
output_folder = "/home/ubuntu/MasterThesis/yunus_resized"
target_bbox_height = 40  # Target median bounding box height
os.makedirs(output_folder, exist_ok=True)

# Dictionary to store adjusted bounding boxes with image names
adjusted_bboxes_dict = {}
adjusted_image_sizes_dict = {}

# Process all images 
for img_name, bboxes in bboxes_per_img.items():
    input_path = os.path.join(input_folder, img_name)
    output_path = os.path.join(output_folder, img_name)

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        continue

    image = cv2.imread(input_path)
    if image is None:
        print(f"Failed to load image: {input_path}")
        continue

    # resize img and adjust bboxes
    resized_image, adjusted_bboxes = resize_image_and_boxes(image, bboxes, target_bbox_height, final_size=(1024,1024))
    if resized_image is None:
        continue

    # save resized image
    cv2.imwrite(output_path, resized_image)
    print(f"Processed and saved: {output_path}")

    # adjusted bboxes and img dimensions
    adjusted_bboxes_dict[img_name] = adjusted_bboxes # image names to bounding boxes
    adjusted_image_sizes_dict[img_name] = resized_image.shape[:2]  # names to new dimensions

print("Adjusted bounding boxes dictionary:", list(adjusted_bboxes_dict.items())[:2])
print("Image sizes dictionary:", list(adjusted_image_sizes_dict.items())[:2])

#endregion

#region Getting Summary Statistics on RESIZED Image Bounding Boxes

stats_per_image_resized = summarize_bounding_boxes(adjusted_bboxes_dict)
print(stats_per_image_resized.head(10))
print(stats_per_image_resized[['median_height', 'num_bboxes', 'max_height', 'min_height', 'mean_height', 'std_height']].head(100))

# bbox height varies between 17 and 58 for these 10 images. Median height is 40 for 8 images, 8.5 for 2 images. Std vary, but all are between 3.5 and7.2. 
# bbox width varies between 10 and 143 for these 10 images. Median width varies between 37 and 54. Std vary between 11.5 and 20.


stats_all_resized = summarize_all_bounding_boxes(adjusted_bboxes_dict)
print(stats_all_resized)
# Summary for all images after resizing
# {'total_bboxes': 54999, 'median_width': 43.0, 'median_height': 38.0, 'mean_width': 45.001600029091435, 'mean_height': 37.4559719267623, 
# 'std_width': 17.137421865308585, 'std_height': 7.441715974103253, 'min_width': 7, 'max_width': 298, 'min_height': 8, 'max_height': 96}

# Median height is at 38 (with mean 37) for all images with max 96 and min 8. Std is at 7.
# Median width is at 43 (with mean at 45) for all images with max value of 298 and min 7. Std is at 17 for width.

#endregion


########## Reordering Bounding Boxes ##########

#region Coming up with five groups of equal images based on bounding boxes

def cluster_images_by_height(stats_df, num_clusters=5):
    """
    Cluster images into groups based on bounding box height statistics using K-Means.

    Args:
        stats_df (pd.DataFrame): DataFrame containing bounding box statistics per image.
        num_clusters (int): Number of clusters to create (default: 5).

    Returns:
        pd.DataFrame: Updated DataFrame with cluster assignments.
        KMeans: Trained K-Means model.
    """
    # which features are relevant?
    features = ["median_height", "mean_height", "min_height", "max_height", "num_bboxes"]
    height_data = stats_df[features]

    # Normalize
    scaler = StandardScaler()
    height_data_normalized = scaler.fit_transform(height_data)

    # apply kmeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=555)
    stats_df["cluster"] = kmeans.fit_predict(height_data_normalized)

    return stats_df, kmeans

num_clusters = 3
clustered_stats, kmeans_model = cluster_images_by_height(stats_per_image_resized, num_clusters=num_clusters)

print(clustered_stats[['cluster', "median_height", "mean_height", "min_height", "max_height", "num_bboxes"]].head())
cluster_mapping = clustered_stats.set_index("image")["cluster"].to_dict()
# Look at clusters, find pattern
# bigger pictures (cluster 1), smaller sorting value
# smaller pictures (cluster 0), higher sorting value
# pictures with very small bbox height (cluster 2) get smallest sorting value

# add cluster information to adjusted bounding boxes dict
def augment_bboxes_with_clusters(bboxes_dict, clustering_stats):
    """
    Augment the bounding boxes dictionary with cluster information.

    Args:
        bboxes_dict (dict): Original dictionary with image names as keys and bounding boxes as values.
        clustering_stats (pd.DataFrame): DataFrame containing image names and their assigned clusters.

    Returns:
        dict: Augmented dictionary with cluster information.
    """
    # mapping from image names to clusters: dict
    cluster_mapping = clustering_stats.set_index("image")["cluster"].to_dict()

    # include cluster info
    augmented_dict = {}
    for img_name, bboxes in bboxes_dict.items():
        cluster = cluster_mapping.get(img_name, -1)  # -1 if not found
        augmented_dict[img_name] = {"bboxes": bboxes, "cluster": cluster}

    return augmented_dict

adjusted_bboxes_dict_clusters = augment_bboxes_with_clusters(adjusted_bboxes_dict, clustered_stats)

first_key = next(iter(adjusted_bboxes_dict_clusters))
first_value = adjusted_bboxes_dict_clusters[first_key]

print(f"First Key: {first_key}")
print(f"First Value: {first_value}")
#endregion

#region Sort bounding boxes, experiment with different thresholds

#NEW function that also takes into account clusters
def sort_bounding_boxes_with_clusters(bounding_boxes_with_clusters, abz_data, cluster_thresholds):
    """
    Sort bounding boxes with cluster-specific row thresholds.

    Args:
        bounding_boxes_with_clusters (dict): Dictionary with image keys containing bounding boxes and cluster info.
        abz_data (dict): Dictionary with image keys and corresponding abz data.
        cluster_thresholds (dict): Mapping of clusters to row thresholds.

    Returns:
        dict, dict: Sorted bounding boxes and sorted abz data.
    """
    sorted_bboxes = {}
    sorted_abz = {}

    for img_name, data in bounding_boxes_with_clusters.items():
        bboxes = data["bboxes"]
        cluster = data["cluster"]

        # Fetch row for threshold cluster
        row_threshold = cluster_thresholds.get(cluster, 20)  # Default trheshold if cluster not in map

        # pair bboxes and abz
        paired = list(zip(bboxes, abz_data.get(img_name, [])))

        # Sort by top (y-coordinate)
        paired = sorted(paired, key=lambda item: item[0][1]) 

        # Group into rows
        rows = []
        current_row = [paired[0]]
        for item in paired[1:]:
            if abs(item[0][1] - current_row[-1][0][1]) <= row_threshold:
                current_row.append(item)
            else:
                rows.append(current_row)
                current_row = [item]
        if current_row:
            rows.append(current_row)

        # Sort rows by left (x-coordinate)
        sorted_rows = [sorted(row, key=lambda item: item[0][0]) for row in rows]

        # Flatten and separate bbox and abz
        sorted_paired = [item for row in sorted_rows for item in row]
        sorted_bboxes[img_name] = [item[0] for item in sorted_paired]
        sorted_abz[img_name] = [item[1] for item in sorted_paired]

    return sorted_bboxes, sorted_abz

cluster_thresholds = {0: 12, 1: 9, 2: 5}  # cluster thresholds

sorted_bboxes_with_clusters, sorted_abz_with_clusters = sort_bounding_boxes_with_clusters(
    adjusted_bboxes_dict_clusters, abz_dict, cluster_thresholds
)

process_images_in_folder('/Users/irina/PythonProjects/MasterThesis/yunus_resized/val', sorted_bboxes_with_clusters)
process_and_save_images_with_abz("thesis_photos/resized", sorted_bboxes_with_clusters, sorted_abz_with_clusters, "thesis_photos/bboxes_ordered/1024x1024/abz")
process_and_save_images("thesis_photos/resized", sorted_bboxes_with_clusters, "thesis_photos/bboxes_ordered/1024x1024/index")
#endregion

#region create new dataset with resized images and bounding boxes
import pandas as pd

def create_new_dataset_with_sorted_abz(df, sorted_bboxes_dict, sorted_abz_dict, image_sizes_dict):
    """
    Create a new dataset with resized image dimensions, adjusted bounding boxes, and sorted abz.

    Args:
        df (pd.DataFrame): Original dataset containing image metadata (img name in right bucket).
        sorted_bboxes_dict (dict): Dictionary with sorted bounding boxes per image.
                                   Format: {image_name: [[x, y, w, h], ...]}.
        sorted_abz_dict (dict): Dictionary with sorted abz values per image.
                                Format: {image_name: [abz1, abz2, ...]}.
        image_sizes_dict (dict): Dictionary with updated image dimensions per image.
                                 Format: {image_name: (height, width)}. 1024x1024 for all images

    Returns:
        pd.DataFrame: New dataset with resized metadata and sorted abz.
    """
    new_rows = []

    for _, row in df.iterrows():
        img_name = row["img_name"]

        bboxes = sorted_bboxes_dict.get(img_name, [])
        height, width = image_sizes_dict.get(img_name, (None, None))
        abz_values = sorted_abz_dict.get(img_name, [])

        # alignment test bboxes and abz values
        if len(bboxes) != len(abz_values):
            print(f"Warning: Mismatch in bbox and abz for {img_name}. Skipping.")
            continue

        # Calculate areas
        areas = [bbox[2] * bbox[3] for bbox in bboxes]  # width * height

        # Create new row with updated data
        new_row = {
            "img_name": img_name,
            "height": height,
            "width": width,
            "bbox": bboxes,
            "area": areas,
            "abz": abz_values,  
        }
        new_rows.append(new_row)

    return pd.DataFrame(new_rows)

# Opening the "old" datasets
with open(train_data_path, 'r') as f:
    train_img_data = pd.DataFrame(json.load(f))

with open(val_data_path, 'r') as f:
    val_img_data = pd.DataFrame(json.load(f))


df_train_resized = create_new_dataset_with_sorted_abz(train_img_data, sorted_bboxes_with_clusters, sorted_abz_with_clusters, adjusted_image_sizes_dict)
df_val_resized = create_new_dataset_with_sorted_abz(val_img_data, sorted_bboxes_with_clusters, sorted_abz_with_clusters, adjusted_image_sizes_dict)

# Save the new dataset - RESIZED
df_train_resized.to_json("/home/ubuntu/MasterThesis/code/yunus_data/df_train_resized.json", orient="records", indent=2)
df_val_resized.to_json("/home/ubuntu/MasterThesis/code/yunus_data/df_val_resized.json", orient="records", indent=2)

print(df_train_resized.columns)
#endregion

#region Splitting resized images in new train and test set

train_resized_path = '/Users/irina/PythonProjects/MasterThesis/code/yunus_data/df_train_resized.json'
val_resized_path = '/Users/irina/PythonProjects/MasterThesis/code/yunus_data/df_val_resized.json'

with open(train_resized_path, 'r') as f:
    df_train_resized = pd.DataFrame(json.load(f))

with open(val_resized_path, 'r') as f:
    df_val_resized = pd.DataFrame(json.load(f))


def organize_images_by_dataset(resized_folder, train_dataset, val_dataset, output_base_folder):
    """
    Organize images into train and validation folders based on the datasets.

    Args:
        resized_folder (str): Path to the folder containing resized images.
        train_dataset (pd.DataFrame): DataFrame containing train dataset metadata.
        val_dataset (pd.DataFrame): DataFrame containing validation dataset metadata.
        output_base_folder (str): Base output folder to store organized images.
    """
    train_output_folder = os.path.join(output_base_folder, "train")
    val_output_folder = os.path.join(output_base_folder, "val")

    # copy images based on dataset
    def copy_images(dataset, destination_folder):
        for img_name in dataset["img_name"]:
            src_path = os.path.join(resized_folder, img_name)
            dest_path = os.path.join(destination_folder, img_name)

            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
                print(f"Copied: {src_path} to {dest_path}")
            else:
                print(f"Image not found: {src_path}")

    # Organize train and validation images
    copy_images(train_dataset, train_output_folder)
    copy_images(val_dataset, val_output_folder)

resized_folder = "/home/ubuntu/MasterThesis/yunus_resized"
output_base_folder = "/home/ubuntu/MasterThesis/yunus_resized"

organize_images_by_dataset(resized_folder, df_train_resized, df_val_resized, output_base_folder)



# Testing if everything worked with bounding boxes at example of validation dataset.
process_images_in_folder('yunus_resized/val', bounding_boxes=sorted_bboxes_with_clusters)
# Looks okay - not perfect, but improved.



#endregion

#region Getting Summary Statistics for Train and Test Set on REORDERED Bounding Boxes - check if same!
bboxes_train_ordered = df_train_resized.set_index("img_name")["bbox"].to_dict()
bboxes_val_ordered = df_val_resized.set_index('img_name')['bbox'].to_dict()

# Summary Statistics Training Data
stats_all_ordered_train = summarize_all_bounding_boxes(bboxes_train_ordered)
print(stats_all_ordered_train)
# Summary for all images after resizing
# {'total_bboxes': 46595, 'median_width': 43.0, 'median_height': 38.0, 'mean_width': 45.01899345423329, 'mean_height': 37.38042708445112, 
# 'std_width': 17.14644459048275, 'std_height': 7.4624941946462995, 'min_width': 7, 'max_width': 298, 'min_height': 8, 'max_height': 96}

# Median height is at 38 (with mean 37) for all training images with max 96 and min 8. Std is at 7.5.
# Median width is at 43 (with mean at 45) for all training images with max value of 298 and min 7. Std is at 17 for width.

stats_all_ordered_val = summarize_all_bounding_boxes(bboxes_val_ordered)
print(stats_all_ordered_val)

# {'total_bboxes': 8404, 'median_width': 43.0, 'median_height': 39.0, 'mean_width': 44.90516420752023, 'mean_height': 37.87482151356497, 
# 'std_width': 17.08698876733006, 'std_height': 7.311296640896795, 'min_width': 9, 'max_width': 199, 'min_height': 10, 'max_height': 75}

# Median height is 39 (mean 37) for all validation images with max 74 and min 10. Std is at 7.3.
# Median width is at 43 (mean 45) for all validation images with max 199 and min 9. Std is 17 for width.
#endregion

#region check if resizing and ordering has worked, i.e. the abz values are still same
with open(val_resized_data_path, 'r') as f:
    df_val_resized = pd.DataFrame(json.load(f))

with open(val_data_path, 'r') as g:
    df_val = pd.DataFrame(json.load(g))

print(df_val_resized.columns)
print(df_val_resized.head())
print(type(df_val_resized['bbox']))
print(df_val_resized['bbox'][1])
print(df_val_resized[['height', 'width']].head())



# create dictionaries ensuring that the lists are flattened, not nested one level too deep
bbox_resized_dict = df_val_resized.groupby('img_name')['bbox'].apply(lambda x: [item for sublist in x for item in sublist]).to_dict()
abz_resized_dict = df_val_resized.groupby('img_name')['abz'].apply(lambda x: [item for sublist in x for item in sublist]).to_dict()
bbox_dict = df_val.groupby('img_name')['bbox'].apply(lambda x: [item for sublist in x for item in sublist]).to_dict()
abz_dict = df_val.groupby('img_name')['abz'].apply(lambda x: [item for sublist in x for item in sublist]).to_dict()

print("Sample bbox_dict:", list(bbox_dict.items())[:1])
print("Sample abz_dict:", list(abz_dict.items())[:1])

save_images_with_abz('/home/ubuntu/MasterThesis/thesis_photos/resized', bbox_resized_dict, abz_resized_dict, '/home/ubuntu/MasterThesis/thesis_photos/resized/bboxes_abz')
save_images_with_abz('/home/ubuntu/MasterThesis/thesis_photos/original', bbox_dict, abz_dict, '/home/ubuntu/MasterThesis/thesis_photos/original/bboxes_abz')

# Same ABZ Values for resized and original images
#endregion

########## Saving in new Train and Validation set: Resized ##########

#region Add category_id to df_train_resized nad df_val_resized

# Loading in classes (from classes.txt file)
categories = ['ABZ13', 'ABZ579', 'ABZ480', 'ABZ70', 'ABZ597', 'ABZ342', 'ABZ461', 'ABZ381', 'ABZ61', 'ABZ1', 'ABZ142', 'ABZ318', 'ABZ231', 'ABZ75', 'ABZ449', 'ABZ533', 'ABZ354', 'ABZ139', 'ABZ545', 'ABZ536', 'ABZ330', 'ABZ308', 'ABZ86', 'ABZ328', 'ABZ214', 'ABZ73', 'ABZ15', 'ABZ295', 'ABZ296', 'ABZ68', 'ABZ55', 'ABZ69', 'ABZ537', 'ABZ371', 'ABZ5', 'ABZ151', 'ABZ411', 'ABZ457', 'ABZ335', 'ABZ366', 'ABZ324', 'ABZ396', 'ABZ206', 'ABZ99', 'ABZ84', 'ABZ353', 'ABZ532', 'ABZ58', 'ABZ384', 'ABZ376', 'ABZ59', 'ABZ334', 'ABZ74', 'ABZ383', 'ABZ589', 'ABZ144', 'ABZ586', 'ABZ7', 'ABZ97', 'ABZ211', 'ABZ399', 'ABZ52', 'ABZ145', 'ABZ343', 'ABZ367', 'ABZ212', 'ABZ78', 'ABZ85', 'ABZ319', 'ABZ207', 'ABZ115', 'ABZ465', 'ABZ570', 'ABZ322', 'ABZ331', 'ABZ38', 'ABZ427', 'ABZ279', 'ABZ112', 'ABZ79', 'ABZ80', 'ABZ60', 'ABZ535', 'ABZ142a', 'ABZ314', 'ABZ232', 'ABZ554', 'ABZ312', 'ABZ172', 'ABZ128', 'ABZ6', 'ABZ595', 'ABZ230', 'ABZ167', 'ABZ12', 'ABZ306', 'ABZ331e+152i', 'ABZ339', 'ABZ134', 'ABZ575', 'ABZ401', 'ABZ313', 'ABZ472', 'ABZ441', 'ABZ62', 'ABZ111', 'ABZ468', 'ABZ148', 'ABZ397', 'ABZ104', 'ABZ147', 'ABZ455', 'ABZ471', 'ABZ412', 'ABZ2', 'ABZ440', 'ABZ101', 'ABZ538', 'ABZ72', 'ABZ298', 'ABZ143', 'ABZ437', 'ABZ393', 'ABZ483', 'ABZ94', 'ABZ559', 'ABZ565', 'ABZ87', 'ABZ138', 'ABZ50', 'ABZ191', 'ABZ152', 'ABZ124', 'ABZ205', 'ABZ398', 'ABZ9', 'ABZ126', 'ABZ164', 'ABZ195', 'ABZ307', 'ABZ598a']

# Adding new category_id column df_train_resized and df_val_resized
abz_to_category_id = {abz: idx for idx, abz in enumerate(categories)}
df_train_resized["category_id"] = df_train_resized["abz"].apply(lambda abz_list: [abz_to_category_id[abz] for abz in abz_list])
df_val_resized["category_id"] = df_val_resized["abz"].apply(lambda abz_list: [abz_to_category_id[abz] for abz in abz_list])

df_train_resized.to_json("/home/ubuntu/MasterThesis/code/yunus_data/df_train_resized.json", orient="records", indent=2)
df_val_resized.to_json("/home/ubuntu/MasterThesis/code/yunus_data/df_val_resized.json", orient="records", indent=2)
#endregion


#region Next Step: Bildverarbeitung
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_image(filepath, as_gray=False):
    """Load an image from the specified filepath."""
    if as_gray:
        return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return cv2.imread(filepath, cv2.IMREAD_COLOR)

def apply_grayscale(image):
    """Convert an image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

def apply_histogram_equalization(image, clip_limit=2.0, grid_size=8):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    return clahe.apply(image)

def apply_adaptive_threshold(image, block_size=11, C=2):
    """Apply adaptive thresholding to an image."""
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
    )

def remove_shadows(image, blur_kernel_size=31):
    """Remove shadows by subtracting a blurred version of the image."""
    blurred_image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)
    shadow_removed = cv2.subtract(image, blurred_image)
    return cv2.normalize(shadow_removed, None, 0, 255, cv2.NORM_MINMAX)

def morphological_operations(image, kernel_size=5, operation_type="dilation"):
    """Apply morphological transformations to an image."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if operation_type == "dilation":
        return cv2.dilate(image, kernel, iterations=1)
    elif operation_type == "erosion":
        return cv2.erode(image, kernel, iterations=1)
    elif operation_type == "opening":
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation_type == "closing":
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError("Invalid morphological operation type.")

def process_images_with_methods(test_folder, output_folder):
    """Process each image in the test folder with various methods using low and high parameters."""
    methods = {
        "histogram_equalization": {
            "low": (1.0, 4),
            "high": (4.0, 16),
            "function": apply_histogram_equalization,
        },
        "adaptive_threshold": {
            "low": (7, 1),
            "high": (21, 5),
            "function": apply_adaptive_threshold,
        },
        "shadow_removal": {
            "high": 75,
            "function": remove_shadows,
        },
        "morphological_operations": {
            "low": (3, "erosion"),
            "function": morphological_operations,
        },
    }

    # Create output folders for each method and parameter
    for method in methods.keys():
        if "low" in methods[method]:
            os.makedirs(os.path.join(output_folder, f"{method}_low"), exist_ok=True)
        if "high" in methods[method]:
            os.makedirs(os.path.join(output_folder, f"{method}_high"), exist_ok=True)

    # process each image
    for filename in os.listdir(test_folder):
        filepath = os.path.join(test_folder, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image = load_image(filepath, as_gray=True)

            for method, params in methods.items():
                # Low parameters
                if "low" in params:
                    if isinstance(params["low"], tuple):
                        low_result = params["function"](image, *params["low"])
                    else:
                        low_result = params["function"](image, params["low"])
                    low_output_path = os.path.join(output_folder, f"{method}_low", filename)
                    cv2.imwrite(low_output_path, low_result)

                # High parameters
                if "high" in params:
                    if isinstance(params["high"], tuple):
                        high_result = params["function"](image, *params["high"])
                    else:
                        high_result = params["function"](image, params["high"])
                    high_output_path = os.path.join(output_folder, f"{method}_high", filename)
                    cv2.imwrite(high_output_path, high_result)

                print(f"Processed {filename} with {method} (low and/or high parameters).")

test_folder = "yunus_photos/train2017/"
output_folder = "yunus_processed/train/"
process_images_with_methods(test_folder, output_folder)
#endregion

#region trying adaptive Tresholding in three different ways
# Define the load_image function
def load_image(filepath, as_gray=True):
    """Load an image from the specified filepath."""
    if as_gray:
        return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return cv2.imread(filepath, cv2.IMREAD_COLOR)


def process_images_with_thresholding(test_folder, output_folder):
    """
    Process each image in the test folder with simple, adaptive mean, and adaptive Gaussian thresholding.

    Args:
        test_folder (str): Path to the folder containing test images.
        output_folder (str): Path to the folder where processed images will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Create subfolders for each thresholding method
    methods = ["simple", "adaptive_mean", "adaptive_gaussian"]
    for method in methods:
        os.makedirs(os.path.join(output_folder, method), exist_ok=True)

    # Process each image
    for filename in os.listdir(test_folder):
        filepath = os.path.join(test_folder, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image = load_image(filepath, as_gray=True)

            # Simple thresholding
            _, simple_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            simple_output_path = os.path.join(output_folder, "simple", filename)
            cv2.imwrite(simple_output_path, simple_thresh)

            # Adaptive mean thresholding
            adaptive_mean_thresh = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
            )
            adaptive_mean_output_path = os.path.join(output_folder, "adaptive_mean", filename)
            cv2.imwrite(adaptive_mean_output_path, adaptive_mean_thresh)

            # Adaptive Gaussian thresholding
            adaptive_gaussian_thresh = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            adaptive_gaussian_output_path = os.path.join(output_folder, "adaptive_gaussian", filename)
            cv2.imwrite(adaptive_gaussian_output_path, adaptive_gaussian_thresh)

            print(f"Processed {filename} with all thresholding techniques.")

# Paths
test_folder = '/Users/irina/PythonProjects/MasterThesis/thesis_photos/resized/'
output_folder = '/Users/irina/PythonProjects/MasterThesis/thesis_photos/resized/'

# Run the function
process_images_with_thresholding(test_folder, output_folder)

#endregion

#region Applying adaptive mean thresholding to train and validation data

import cv2
import os

def apply_adaptive_mean_thresholding(input_folder, output_folder):
    """
    Apply adaptive mean thresholding to all images in the input folder and save the results in the output folder.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where processed images will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)

        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Failed to load image: {filepath}")
                continue

            adaptive_mean_thresh = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
            )

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, adaptive_mean_thresh)
            print(f"Processed and saved: {output_path}")

input_train = "/Users/irina/PythonProjects/MasterThesis/yunus_resized/train/"
output_train = "/Users/irina/PythonProjects/MasterThesis/yunus_processed/train/"
input_val = "/Users/irina/PythonProjects/MasterThesis/yunus_resized/val/"
output_val = "/Users/irina/PythonProjects/MasterThesis/yunus_processed/val/"

apply_adaptive_mean_thresholding(input_train, output_train)
apply_adaptive_mean_thresholding(input_val, output_val)




#endregion


########## Data Augmentation ##########
#region Resized and Rotate original cutted images 

import cv2
import numpy as np
import os

def resize_and_rotate_image(image, rotation_angle, target_bbox_height, max_image_size=(1024, 1024), final_size=(1024, 1024)):
    """
    Resize an image based on bounding box height, apply rotation, and apply additional resizing and padding.

    Args:
        image (np.array): Input image.
        rotation_angle (float): Rotation angle in degrees.
        target_bbox_height (int): Target median bounding box height.
        max_image_size (tuple): Maximum allowable image size (height, width).
        final_size (tuple): Final quadratic size for padding.

    Returns:
        np.array: Final resized, rotated, and padded image.
    """
    # rotation matrix
    original_h, original_w = image.shape[:2]
    center = (original_w // 2, original_h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    # Compute new image dimensions after rotation
    cos_theta = np.abs(rotation_matrix[0, 0])
    sin_theta = np.abs(rotation_matrix[0, 1])
    new_w = int(original_h * sin_theta + original_w * cos_theta)
    new_h = int(original_h * cos_theta + original_w * sin_theta)

    # Adjust rotation matrix for translation
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # rotate
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))

    # rsize rotated image based on target bounding box height
    scaling_factor = target_bbox_height / original_h  
    resized_h, resized_w = int(new_h * scaling_factor), int(new_w * scaling_factor)
    resized_image = cv2.resize(rotated_image, (resized_w, resized_h))

    # image < max_size
    max_h, max_w = max_image_size
    if resized_h > max_h or resized_w > max_w:
        scaling_factor = min(max_w / resized_w, max_h / resized_h)
        resized_h, resized_w = int(resized_h * scaling_factor), int(resized_w * scaling_factor)
        resized_image = cv2.resize(resized_image, (resized_w, resized_h))

    # pad
    final_h, final_w = final_size
    pad_top = max((final_h - resized_h) // 2, 0)
    pad_bottom = max(final_h - resized_h - pad_top, 0)
    pad_left = max((final_w - resized_w) // 2, 0)
    pad_right = max(final_w - resized_w - pad_left, 0)

    padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return padded_image

input_folder = "/home/ubuntu/MasterThesis/cutted_images/"
output_folder = "/home/ubuntu/MasterThesis/yunus_rotated"
target_bbox_height = 40 
rotation_angles = [-5, 5] 
os.makedirs(output_folder, exist_ok=True)

for img_name in os.listdir(input_folder):
    input_path = os.path.join(input_folder, img_name)

    if not os.path.isfile(input_path) or not img_name.endswith(".jpg"):
        continue

    image = cv2.imread(input_path)
    if image is None:
        print(f"Failed to load image: {input_path}")
        continue

    resized_image = resize_and_rotate_image(image, 0, target_bbox_height) 
    cv2.imwrite(os.path.join(output_folder, img_name), resized_image)
    print(f"Processed and saved original: {img_name}")

    for angle in rotation_angles:
        rotated_image = resize_and_rotate_image(image, angle, target_bbox_height)
        angle_str = f"{'+' if angle > 0 else ''}{angle:02d}"  # Format angle as +05 or -05
        rotated_name = f"{img_name[:-4]}_{angle_str}.jpg"
        cv2.imwrite(os.path.join(output_folder, rotated_name), rotated_image)
        print(f"Processed and saved rotated: {rotated_name}")

#endregion

#region Adjusting dataframe
# Opening the "old" datasets
train_data_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_train_resized.json'
val_data_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_val_resized.json'

with open(train_data_path, 'r') as f:
    df_train_resized = pd.DataFrame(json.load(f))

with open(val_data_path, 'r') as f:
    df_val_resized = pd.DataFrame(json.load(f))

print(df_val_resized.columns)
print(df_train_resized.columns)

# Creating new datasets with augmented data

df_train_resized_plus_05 = df_train_resized.copy()
df_train_resized_minus_5 = df_train_resized.copy()

# Adjust image names for +5 and -5 rotations
df_train_resized_plus_05['img_name'] = df_train_resized_plus_05['img_name'].str.replace(
    r'(.jpg)$', r'_+05\1', regex=True
)
df_train_resized_minus_5['img_name'] = df_train_resized_minus_5['img_name'].str.replace(
    r'(.jpg)$', r'_-5\1', regex=True
)

print(df_train_resized_minus_5['img_name'])
print(df_train_resized_plus_05['img_name'].head())


# Concatenate the original, +5, and -5 datasets
df_train_aug_rot = pd.concat(
    [df_train_resized, df_train_resized_plus_05, df_train_resized_minus_5],
    ignore_index=True
)

def save_to_json(df, folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    print(file_path)
    data = df.to_dict(orient='records')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

data_folder = '/home/ubuntu/MasterThesis/code/yunus_data/'
save_to_json(df_train_aug_rot, data_folder, 'df_train_aug_rot.json')

#endregion

#region Try with already resized images and bounding boxes
def augment_images_with_rotation(input_folder, output_folder, df_train, rotation_angles=(5, -5)):
    """
    Augment images by rotating them and updating bounding boxes.

    Args:
        input_folder (str): Path to the folder containing original images.
        output_folder (str): Path to save augmented images.
        df_train (pd.DataFrame): DataFrame containing metadata for training images.
        rotation_angles (tuple): Rotation angles in degrees (e.g., (5, -5)).

    Returns:
        pd.DataFrame: Updated DataFrame with augmented images and bounding boxes.
    """
    os.makedirs(output_folder, exist_ok=True)
    augmented_rows = []

    for _, row in df_train.iterrows():
        img_name = row['img_name']
        img_path = os.path.join(input_folder, img_name)

        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_name}")
            continue

        height, width = row['height'], row['width']
        bboxes = row['bbox']  

        original_path = os.path.join(output_folder, img_name)
        cv2.imwrite(original_path, image)
        augmented_rows.append(row.to_dict())  # adding original unchanged

        # rotate for each angle
        for angle in rotation_angles:
            # Rotate the image
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderValue=(0, 0, 0))

            # Rotate the bounding boxes
            rotated_bboxes = []
            for bbox in bboxes:
                x, y, w, h = bbox
                corners = np.array([
                    [x, y],
                    [x + w, y],
                    [x, y + h],
                    [x + w, y + h]
                ])
                corners = np.hstack([corners, np.ones((4, 1))])  # Add column for affine transformation
                rotated_corners = np.dot(rotation_matrix, corners.T).T
                x_min, y_min = rotated_corners[:, 0].min(), rotated_corners[:, 1].min()
                x_max, y_max = rotated_corners[:, 0].max(), rotated_corners[:, 1].max()
                rotated_bboxes.append([int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)])

            # Recalculate areas
            rotated_areas = [w * h for _, _, w, h in rotated_bboxes]

            # reuse metadata and modify necessary columns
            rotated_row = row.to_dict()  
            rotated_row['img_name'] = os.path.splitext(img_name)[0] + f"_{angle:+0d}.jpg"
            rotated_row['bbox'] = rotated_bboxes
            rotated_row['area'] = rotated_areas

            rotated_img_path = os.path.join(output_folder, rotated_row['img_name'])
            cv2.imwrite(rotated_img_path, rotated_image)

            augmented_rows.append(rotated_row)

    augmented_df = pd.DataFrame(augmented_rows)
    return augmented_df

# Paths
input_folder = "/home/ubuntu/MasterThesis/yunus_resized/train_old/"
output_folder = "/home/ubuntu/MasterThesis/yunus_rotated"
train_resized_data_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_train_resized.json'

with open(train_resized_data_path, 'r') as f:
    df_train_resized = pd.DataFrame(json.load(f))
print(df_train_resized.columns)


df_train_aug_rotation = augment_images_with_rotation(input_folder, output_folder, df_train_resized)
df_train_aug_rotation.to_json("/home/ubuntu/MasterThesis/code/yunus_data/df_train_aug_newrotation.json", orient="records", indent=4)

# Check if rotation with bboxes worked!! 
bbox_aug_dict = df_train_aug_rotation.groupby('img_name')['bbox'].apply(lambda x: [item for sublist in x for item in sublist]).to_dict()
abz_aug_dict = df_train_aug_rotation.groupby('img_name')['abz'].apply(lambda x: [item for sublist in x for item in sublist]).to_dict()
bbox_resized_train_dict = df_train_resized.groupby('img_name')['bbox'].apply(lambda x: [item for sublist in x for item in sublist]).to_dict()
abz_resized_train_dict = df_train_resized.groupby('img_name')['abz'].apply(lambda x: [item for sublist in x for item in sublist]).to_dict()
save_images_with_abz('/home/ubuntu/MasterThesis/thesis_photos/train/original', bounding_boxes=bbox_resized_train_dict, abz_data=abz_resized_train_dict, output_folder='/home/ubuntu/MasterThesis/thesis_photos/train/original/bboxes_abz')
save_images_with_abz('/home/ubuntu/MasterThesis/thesis_photos/train/augmented_rotated', bounding_boxes=bbox_aug_dict, abz_data=abz_aug_dict, output_folder='/home/ubuntu/MasterThesis/thesis_photos/train/augmented_rotated/bboxes_abz')

#endregion

#region Create validation dataset for finetuning 
df_aug_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_train_aug_newrotation.json'

with open(df_aug_path, 'r') as f:
    df_aug = pd.DataFrame(json.load(f))

df_aug_shuffled = df_aug.sample(frac=1, random_state=1234).reset_index(drop=True)
    
# Updated train-validation-test split function
def train_val_split(df):
    train_split = int(0.8 * len(df))

    df_train = df[:train_split]
    df_val = df[train_split:]

    return df_train, df_val

df_train, df_val= train_val_split(df_aug_shuffled)

df_train.to_json('/home/ubuntu/MasterThesis/code/yunus_data/df_train_aug.json', orient="records", indent=4)
df_val.to_json('/home/ubuntu/MasterThesis/code/yunus_data/df_val_aug.json', orient='records', indent = 4)

# Move the images to the right folders

def create_image_datasets(image_folder, train_df, val_df, output_folder):
    """
    Create sub-folders for training and validation images based on splits.

    Args:
        image_folder (str): Path to the folder containing all images.
        train_df (pd.DataFrame): DataFrame containing the training split.
        val_df (pd.DataFrame): DataFrame containing the validation split.
        output_folder (str): Path to the output folder where sub-folders will be created.

    Returns:
        Nothing
    """
    #  output directories for train and validation images
    train_output_folder = os.path.join(output_folder, "train")
    val_output_folder = os.path.join(output_folder, "validation")
    os.makedirs(train_output_folder, exist_ok=True)
    os.makedirs(val_output_folder, exist_ok=True)

    # Copy training images
    for img_name in train_df["img_name"]:
        src_path = os.path.join(image_folder, img_name)
        dest_path = os.path.join(train_output_folder, img_name)
        shutil.copy(src_path, dest_path)

    # Copy validation images
    for img_name in val_df["img_name"]:
        src_path = os.path.join(image_folder, img_name)
        dest_path = os.path.join(val_output_folder, img_name)
        shutil.copy(src_path, dest_path)


    print("Finished!")


image_folder = "/home/ubuntu/MasterThesis/yunus_aug_rotation"  
output_folder = "/home/ubuntu/MasterThesis/yunus_aug_rotation"    

create_image_datasets(image_folder, df_train, df_val, output_folder)

#endregion

#region Use Adaptive Thresholding with augmented data

def apply_adaptive_gaussian_thresholding(input_folder, output_folder):
    """
    Apply adaptive Gaussian thresholding to all images in the input folder and save the results in the output folder.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where processed images will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)

        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            if image is None or image.size == 0:
                print(f"Warning: Failed to load image or image is empty: {filepath}")
                continue

            try:
                adaptive_gaussian_thresh = cv2.adaptiveThreshold(
                    image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )

                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, adaptive_gaussian_thresh)
                print(f"Processed and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

# Paths
input_train = "/home/ubuntu/MasterThesis/yunus_aug_rotation/train/"
output_train = "/home/ubuntu/MasterThesis/yunus_aug_rotation_adapt_thresh/train"
input_val = "/home/ubuntu/MasterThesis/yunus_aug_rotation/validation"
output_val = "/home/ubuntu/MasterThesis/yunus_aug_rotation_adapt_thresh/val"
input_test = '/home/ubuntu/MasterThesis/yunus_resized/val'
output_test = '/home/ubuntu/MasterThesis/yunus_resized/val/adapt_thresh'

# Run the function
apply_adaptive_gaussian_thresholding(input_train, output_train)
apply_adaptive_gaussian_thresholding(input_val, output_val)
apply_adaptive_gaussian_thresholding(input_test, output_test)

#endregion

#region Try Erosion

def apply_erosion(input_folder, output_folder, kernel_size=3, iterations=1):
    """
    Apply erosion to all images in the input folder and save the results in the output folder.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where processed images will be saved.
        kernel_size (int): Size of the erosion kernel (default is 3x3).
        iterations (int): Number of iterations for the erosion operation (default is 1).

    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)

    # erosion kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)

        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        # skip if img has error
        if image is None or image.size == 0:
            print(f"Warning: Failed to load image or image is empty: {filepath}")
            continue

        try: # erosion
            eroded_image = cv2.erode(image, kernel, iterations=iterations)

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, eroded_image)
            print(f"Processed and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

def apply_dilation(input_folder, output_folder, kernel_size=3, iterations=1):
    """
    Apply dilation to all images in the input folder and save the results in the output folder.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where processed images will be saved.
        kernel_size (int): Size of the dilation kernel (default is 3x3).
        iterations (int): Number of iterations for the dilation operation (default is 1).

    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)

    # dilation kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)


        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        try: # dilation
            dilated_image = cv2.dilate(image, kernel, iterations=iterations)

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, dilated_image)
            print(f"Processed and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {filepath}: {e}")


# Paths
input_train = "/home/ubuntu/MasterThesis/yunus_aug_rotation/train/"
output_train = "/home/ubuntu/MasterThesis/yunus_aug_rotation_erosion/train"
input_val = "/home/ubuntu/MasterThesis/yunus_aug_rotation/validation"
output_val = "/home/ubuntu/MasterThesis/yunus_aug_rotation_erosion/val"
input_test = '/home/ubuntu/MasterThesis/yunus_resized/val'
output_test = '/home/ubuntu/MasterThesis/yunus_resized/val/erosion'

apply_erosion(input_train, output_train, kernel_size=3, iterations=1)
apply_erosion(input_val, output_val, kernel_size=3, iterations = 1)
apply_dilation(input_train, output_train, kernel_size=3, iterations=1)
apply_dilation(input_val, output_val, kernel_size=3, iterations = 1)
apply_erosion(input_test, output_test)
#endregion

# Archive
#region KMeans Clustering to find 5 similar groups
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

def kmeans_group_images_by_height(bounding_boxes, num_clusters=5):
    """
    Group images into clusters based on the median height of bounding boxes using K-Means clustering.

    Args:
        bounding_boxes (dict): Dictionary where keys are image names and values are lists of bounding boxes [x, y, width, height].
        num_clusters (int): Number of clusters to create (default is 5).

    Returns:
        pd.DataFrame: A DataFrame with image names, median heights, and their assigned cluster.
    """
    # median height per image
    image_heights = []
    for img_name, bboxes in bounding_boxes.items():
        heights = [bbox[3] for bbox in bboxes]  
        median_height = np.median(heights) if heights else 0  
        image_heights.append({"image": img_name, "median_height": median_height})

    df = pd.DataFrame(image_heights)

    # prepare for clustering
    X = df["median_height"].values.reshape(-1, 1)  # Reshape to 2D array for K-Means

    # apply clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)  
    df["cluster"] = kmeans.fit_predict(X)  

    return df, kmeans


grouped_images, kmeans_model = kmeans_group_images_by_height(bboxes_per_img, num_clusters=10)
print(grouped_images)

#endregion

#region Copying images for master thesis
img_list_train = ['NBC.4617-0.jpg', 'P336660-0.jpg'] 
img_list_val = ['YBC.9120-0.jpg']

def copy_images(image_list, source_folder, destination_folder):
    for img_name in image_list:
        src_path = os.path.join(source_folder, img_name)
        dest_path = os.path.join(destination_folder, img_name)
        shutil.copy(src_path, dest_path)

source_folder = "/home/ubuntu/MasterThesis/yunus_aug_rotation_dilation/train"
destination_folder = "/home/ubuntu/MasterThesis/thesis_photos/train/erosion"

copy_images(img_list_train, source_folder, destination_folder)

#endregion
