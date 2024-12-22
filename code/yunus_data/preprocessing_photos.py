#region Testing if photos can be cut even more
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd

train_data_path = 'code/yunus_data/df_train.json'
val_data_path = 'code/yunus_data/df_val.json'

with open(train_data_path, 'r') as f:
    train_img_data = pd.DataFrame(json.load(f))
print(train_img_data.columns)
print(len(train_img_data))

with open(val_data_path, 'r') as f:
    val_img_data = pd.DataFrame(json.load(f))

img_data_concat = pd.concat([train_img_data, val_img_data], ignore_index=True)
print(img_data_concat.head())
print(len(img_data_concat))

bboxes_per_img = img_data_concat.set_index("img_name")["bbox"].to_dict()
for key, value in list(bboxes_per_img.items())[:2]:
    print(f"{key}: {value}")


def process_images_in_folder(folder_path, bounding_boxes=None):
    """
    Process all images in a folder to display tablet contours and existing bounding boxes.

    Args:
        folder_path (str): Path to the folder containing images.
        bounding_boxes (dict): Existing bounding boxes for images, where keys are image file names and values are lists of bounding boxes.
    """
    # List all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Load the image
        image = cv2.imread(file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Preprocessing: Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Binarization using Otsu's thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological operations to merge nearby regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        morphed = cv2.dilate(binary, kernel, iterations=2)

        # Edge Detection
        edges = cv2.Canny(morphed, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area and aspect ratio
        filtered_contours = [
            cnt for cnt in contours 
            if cv2.contourArea(cnt) > 5000 and 0.5 < cv2.boundingRect(cnt)[2] / cv2.boundingRect(cnt)[3] < 2
        ]

        # Draw contours and bounding boxes
        display_image = image.copy()
        cv2.drawContours(display_image, filtered_contours, -1, (0, 255, 0), 2)

        if bounding_boxes and file_name in bounding_boxes:
            for bbox in bounding_boxes[file_name]:
                x, y, w, h = bbox
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the original image with contours and bounding boxes
        plt.figure(figsize=(10, 10))
        plt.title(f"Contours and Bounding Boxes for {file_name}")
        plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

# Example usage:
# folder_path = "path_to_images_folder"
# process_images_in_folder(folder_path)

def process_images_in_folder_with_lines(folder_path):
    """
    Process all images in a folder and draw horizontal lines every 500 pixels to indicate potential cutting points.

    Args:
        folder_path (str): Path to the folder containing images.
    """
    # List all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Load the image
        image = cv2.imread(file_path)
        height, width, _ = image.shape

        # Draw horizontal lines every 500 pixels
        display_image = image.copy()
        line_spacing = 500

        for y in range(0, height, line_spacing):
            cv2.line(display_image, (0, y), (width, y), (0, 255, 0), 2)  # Green lines
            cv2.putText(display_image, f"y={y}", (10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)  # Add y-coordinate

        # Display the image with lines
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
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    height, width, _ = image.shape

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Split the image into two parts at the line_y
    top_part = image[:line_y, :]
    bottom_part = image[line_y:, :]

    # Save the two parts
    file_name = os.path.basename(image_path)
    top_path = os.path.join(output_folder, f"{file_name}_top.jpg")
    bottom_path = os.path.join(output_folder, f"{file_name}_bottom.jpg")

    cv2.imwrite(top_path, top_part)
    cv2.imwrite(bottom_path, bottom_part)

    print(f"Saved top part to {top_path}")
    print(f"Saved bottom part to {bottom_path}")

# Define the folder path containing the images
folder_path = "cutted_images/"
output_folder = "extracted_tablets"


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

#endregion


# Extract tablets and save to the output folder
# Use the `process_images_in_folder` function to display bounding boxes
#process_images_in_folder_with_lines(folder_path)
# Extract tablets and save to the output folder
#extract_tablets_with_bounding_boxes(output_folder, bboxes_per_img, output_folder, padding=20)

# Show images with their bounding boxes
#process_images_in_folder(folder_path, bounding_boxes=bboxes_per_img)


#endregion

#region Bildverarbeitung
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

    # Process each image
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

