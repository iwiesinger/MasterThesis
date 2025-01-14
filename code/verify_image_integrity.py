from PIL import Image
import os

def remove_corrupted_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            try:
                with Image.open(os.path.join(directory, filename)) as img:
                    img.verify()  # Verify the image integrity
            except (IOError, SyntaxError):
                print(f"Removing corrupted file: {filename}")
                os.remove(os.path.join(directory, filename))

remove_corrupted_images("path_to_image_directory")

import os
from PIL import Image

def count_images(folder_path):
    image_count = 0
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            with Image.open(file_path) as img:
                img.verify()  # Check if it's a valid image
            image_count += 1
        except Exception:
            continue  # Skip files that are not images
    return image_count

folder_path = "/home/ubuntu/MasterThesis/language_model_photos/" 
print(f"Number of images in the folder: {count_images(folder_path)}")
# Number of images in the folder: 19284



import os

def count_items_in_folder(folder_path):
    return len(os.listdir(folder_path))

folder_path = "/home/ubuntu/MasterThesis/language_model_photos/" 
print(f"Number of items in the folder: {count_items_in_folder(folder_path)}")



import os
from PIL import Image

def find_non_images(folder_path):
    non_image_files = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            with Image.open(file_path) as img:
                img.verify()  # Verify if it's a valid image
        except Exception:
            non_image_files.append(file_name)  # Add non-images to the list
    return non_image_files

folder_path = "/home/ubuntu/MasterThesis/language_model_photos/" 
non_images = find_non_images(folder_path)

if non_images:
    print("Non-image files:")
    for file_name in non_images:
        print(file_name)
else:
    print("No non-image files found.")

'''
Non-image files:
BM.33841.jpg
YBC.4625.jpg
K.2419.jpg
YBC.4189.jpg
BM.45684.jpg
YBC.4188.jpg
BM.47461.jpg
BM.34081.jpg
BM.45697.jpg
BM.34655.jpg
YBC.4642.jpg
BM.45736.jpg
BM.34104.jpg
Rm.716.jpg
BM.38552.jpg
YBC.4644.jpg
BM.47812.jpg
BM.36601.jpg
BM.45840.jpg
BM.45791.jpg
BM.33483.jpg
BM.47447.jpg
BM.40090.jpg
BM.45989.jpg
BM.34685.jpg
MLC.1879.jpg
BM.28825.jpg
BM.47754.jpg
K.2836.jpg
BM.34035.jpg
BM.46336.jpg
BM.38895.jpg
BM.32656.jpg
BM.47456.jpg
BM.34584.jpg
BM.36606.jpg
BM.46566.jpg
BM.47880.jpg
BM.45788.jpg
BM.46338.jpg
BM.45947.jpg
K.2361.jpg
BM.40795.jpg
BM.35564.jpg
BM.41497.jpg
BM.45655.jpg
BM.45652.jpg
BM.41284.jpg
BM.34637.jpg
BM.48518.jpg
YBC.2292.jpg
BM.45795.jpg
BM.46293.jpg
BM.45996.jpg
K.7883.jpg
K.3753.jpg
BM.45800.jpg
BM.47799.jpg
YBC.4643.jpg
K.10422.jpg
K.2133.jpg
BM.36647.jpg
BM.33333.B.jpg
BM.35966.jpg
BM.36066.jpg
BM.34676.jpg
BM.34138.jpg
BM.34092.jpg
BM.45649.jpg
BM.61625.jpg
'''


def delete_non_images(folder_path, non_image_files):
    for file_name in non_image_files:
        file_path = os.path.join(folder_path, file_name)
        os.remove(file_path)  # Delete the file
        print(f"Deleted: {file_name}")

delete_non_images(folder_path, non_images)