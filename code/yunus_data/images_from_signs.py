########## Preparing texts that should be copied with bounding boxes of other images ##########

#region imports
import pandas as pd
import json
import pandas as pd
from PIL import Image, ImageOps
import os
import random

#endregion

df_new_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_new.json'
vocab_path = '/home/ubuntu/MasterThesis/code/yunus_data/vocab.json'

with open(df_new_path, 'r') as f:
    df_new = pd.DataFrame(json.load(f))

print(df_new.columns) # Index(['_id', 'signs', 'tok_signs', 'signs_len', 'X_percent', 'img_name', 'input_ids', 'attention_mask', 'labels', 'tok_len'], dtype='object')
print(len(df_new))

# Preprocess further: Delete BOS, EOS and UNK tokens.
print(df_new['tok_signs'][1])

# exclude BOS, EOS and UNK tokens.
def exclude_tokens(tok_signs):
    return [token for token in tok_signs if token not in {'<BOS>', '<EOS>', '<UNK>'}]

df_new['abz'] = df_new['tok_signs'].apply(exclude_tokens)
print(df_new['abz'].head())



# How many columns have leq 500 tokens? (to fit into image)
df_new['tok_len'] = df_new['abz'].apply(len)
toks_leq500 = df_new[df_new['tok_len']<=500]
print(len(toks_leq500)) # 2325

# Is the whole vocabulary covered by these 2300 observations?
with open(vocab_path, 'r') as f:
    vocab = set(json.load(f))

all_tokens500 = set()
toks_leq500['tok_signs'].apply(lambda x: all_tokens500.update(x))

missing_tokens500 = vocab - all_tokens500
coverage_percentage500 = (1 - len(missing_tokens500)/len(vocab))
print(f'Total vocabulary: {len(vocab)}, \nTokens in Dataset: {len(all_tokens500)}, \nCoverage percentage: {coverage_percentage500}, \nMissing tokens: {len(missing_tokens500)}')
# 7 missing tokens, 94% coverage - good enough for now

import os

# List of already created images
created_images = set(os.listdir(output_folder))
print(len(created_images))

def filter_missing_images(df, created_images):
    remaining_rows = []
    for index, row in df.iterrows():
        base_img_name = row['img_name'].rsplit('.jpg', 1)[0]
        for variation in range(1, num_variations + 1):
            img_name = f"{base_img_name}_art{variation}.jpg"
            if img_name not in created_images:
                remaining_rows.append((index, variation))
    return remaining_rows

# Validate indices in remaining_tasks
invalid_indices = [task[0] for task in remaining_tasks if task[0] >= len(toks_leq500)]
if invalid_indices:
    print(f"Invalid indices found: {invalid_indices}")
remaining_tasks = [(index, variation) for index, variation in remaining_tasks if index < len(toks_leq500)]


remaining_tasks = filter_missing_images(toks_leq500, created_images)
print(f"Remaining images to generate: {len(remaining_tasks)}")
print(type(remaining_tasks))


def resume_generation(toks_leq25, remaining_tasks):
    for index, variation in remaining_tasks:
        row = toks_leq25.iloc[index]
        sequence = row['abz'] 
        base_img_name = row['img_name'].rsplit('.jpg', 1)[0]

        # blank canvas with a black background
        canvas = Image.new("RGB", (canvas_size, canvas_size), "black")

        # Calculate height and width of the sequence
        num_rows = (len(sequence) + letters_per_row - 1) // letters_per_row  # Rows needed
        total_height = num_rows * sign_size
        total_width = min(len(sequence), letters_per_row) * sign_size

        # Centering offsets
        y_start = (canvas_size - total_height) // 2
        x_start = (canvas_size - total_width) // 2

        # placing tokens
        for idx, token in enumerate(sequence):
            row_idx = idx // letters_per_row
            col_idx = idx % letters_per_row
            x_offset = x_start + col_idx * sign_size
            y_offset = y_start + row_idx * sign_size

            # Load and Resize 
            token_folder = os.path.join(input_folder, token)
            if not os.path.exists(token_folder):
                print(f"Warning: Folder for token '{token}' not found. Skipping.")
                continue

            token_images = os.listdir(token_folder)
            if not token_images:
                print(f"Warning: No images found in folder '{token_folder}'. Skipping.")
                continue

            selected_image = random.choice(token_images)
            image_path = os.path.join(token_folder, selected_image)
            sign_img = resize_and_pad(image_path, sign_size)

            # place on canvas
            canvas.paste(sign_img, (x_offset, y_offset))

        # Save the generated image
        new_img_name = f"{base_img_name}_art{variation}.jpg"
        canvas.save(os.path.join(output_folder, new_img_name))

        # Optional: Update progress tracking
        print(f"Generated: {new_img_name}")

resume_generation(toks_leq500, remaining_tasks)


# Configuration
input_folder = "/home/ubuntu/MasterThesis/train_set/"  
output_folder = "/home/ubuntu/MasterThesis/artificial_images"  
os.makedirs(output_folder, exist_ok=True)
canvas_size = 1024
sign_size = 40
letters_per_row = canvas_size // sign_size  # Maximum letters per row (25)
num_variations = 30  # Number of variations per observation

# Function to resize and pad image to 40x40 while preserving aspect ratio
def resize_and_pad(image_path, target_size):
    img = Image.open(image_path).convert("RGB")
    
    # Resize while maintaining aspect ratio
    img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Calculate padding to make the image exactly target_size x target_size
    delta_w = target_size - img.size[0]
    delta_h = target_size - img.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    
    # Expand the image with black padding
    padded_img = ImageOps.expand(img, padding, fill="black")
    return padded_img



# Function to generate synthetic images with centered signs and a black background
def generate_artificial_images_and_dataset(toks_leq25):
    df_art = pd.DataFrame()  
    for index, row in toks_leq25.iterrows():
        sequence = row['abz'] 
        base_img_name = row['img_name'].rsplit('.jpg', 1)[0] 

        for variation in range(num_variations):
            # blank canvas with a black background
            canvas = Image.new("RGB", (canvas_size, canvas_size), "black")

            # Calculate height and width of the sequence
            num_rows = (len(sequence) + letters_per_row - 1) // letters_per_row  # Rows needed
            total_height = num_rows * sign_size
            total_width = min(len(sequence), letters_per_row) * sign_size

            # Centering offsets
            y_start = (canvas_size - total_height) // 2
            x_start = (canvas_size - total_width) // 2

            # placing tokens
            for idx, token in enumerate(sequence):
                row_idx = idx // letters_per_row
                col_idx = idx % letters_per_row
                x_offset = x_start + col_idx * sign_size
                y_offset = y_start + row_idx * sign_size

                # Load and Resize 
                token_folder = os.path.join(input_folder, token)
                if not os.path.exists(token_folder):
                    print(f"Warning: Folder for token '{token}' not found. Skipping.")
                    continue

                token_images = os.listdir(token_folder)
                if not token_images:
                    print(f"Warning: No images found in folder '{token_folder}'. Skipping.")
                    continue

                selected_image = random.choice(token_images)
                image_path = os.path.join(token_folder, selected_image)
                sign_img = resize_and_pad(image_path, sign_size)

                # place on canvas
                canvas.paste(sign_img, (x_offset, y_offset))

            # Save the generated image
            new_img_name = f"{base_img_name}_art{variation + 1}.jpg"
            canvas.save(os.path.join(output_folder, new_img_name))

            # Add a new row to df_art (dataframe of these images)
            new_row = row.copy()
            new_row['img_name'] = new_img_name
            df_art = pd.concat([df_art, pd.DataFrame([new_row])], ignore_index=True)

    return df_art

df_new.columns

df_art = generate_artificial_images_and_dataset(toks_leq500)
print(len(df_art))

df_train_big_aug_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_train_big_aug.json'
with open(df_train_big_aug_path, 'r') as f:
    df_train_big_aug = pd.DataFrame(json.load(f))

print(df_art.columns)
print(df_train_big_aug.columns)

df_big_art_aug = pd.concat([df_big_aug, df_art], ignore_index=True)
df_big_art_aug.to_json("/home/ubuntu/MasterThesis/code/yunus_data/df_train_big_art_aug.json", orient='records', indent=4)  

#region Copy images from one region to other

# Paths
first_folder = "/home/ubuntu/MasterThesis/yunus_random_augment"  # Folder containing newly created images
destination_folder = "/home/ubuntu/MasterThesis/yunus_big_art_random_augment"  # Folder where images will be copied

# Iterate over files in the source folder
for root, _, files in os.walk(first_folder):
    for file in files:
            source_path = os.path.join(root, file)
            destination_path = os.path.join(destination_folder, file)
            shutil.copy(source_path, destination_path)

print(f"All images copied to {destination_folder}")


def count_files_in_folder(folder_path):
    length = len(os.listdir(folder_path))
    return length

count_files_in_folder('/home/ubuntu/MasterThesis/artificial_images')

import os

def count_files_in_subfolders(folder_path):
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):  # Ensure it's a directory
            file_count = len(os.listdir(subfolder_path))
            print(f"Sub-folder: {subfolder}, Number of files: {file_count}")

# Example usage
folder_path = "/home/ubuntu/MasterThesis/train_set"
count_files_in_subfolders(folder_path)

df_train_val_aug_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_train_val_aug.json'

with open(df_train_val_aug_path, 'r') as f:
    df_train_val_aug = pd.DataFrame(json.load(f))

print(len(df_train_val_aug))

print(df_train_val_aug.columns)


df_val_big_aug_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_val_big_aug.json'
with open(df_val_big_aug_path, 'r') as f:
    df_val_big_aug = pd.DataFrame(json.load(f))

print(len(df_val_big_aug))


######## There was an error with the creation of df_art - needs to be created separately

num_variations = 30

# Function to create df_art from df_new
def create_df_art(df_new):
    duplicated_rows = []

    # Loop through each row in df_new
    for _, row in df_new.iterrows():
        base_img_name = row['img_name'].rsplit('.jpg', 1)[0]
        
        # Duplicate the row for each variation
        for variation in range(1, num_variations + 1):
            new_row = row.copy()
            new_row['img_name'] = f"{base_img_name}_art{variation}.jpg"
            duplicated_rows.append(new_row)

    # Create the new dataframe
    df_art = pd.DataFrame(duplicated_rows)
    return df_art

# Create df_art
df_art = create_df_art(df_new)
df_art.to_json('/home/ubuntu/MasterThesis/code/yunus_data/df_art.json', orient='records', indent=4)
print(len(df_art))

image_folder = '/home/ubuntu/MasterThesis/artificial_images'

# Get a set of all existing image filenames
existing_images = set(os.listdir(image_folder))

# Filter df_art to only include rows with img_name matching existing images
df_art_filtered = df_art[df_art['img_name'].isin(existing_images)]
print(len(df_art_filtered)) # 65100 as images

df_art_filtered.to_json('/home/ubuntu/MasterThesis/code/yunus_data/df_art.json', orient='records', indent=4)


######## Vonvert images to greyscale and apply adaptive thresholding

import cv2
import os
import cv2
import os
from tqdm import tqdm


# folder 
output_folder = "/home/ubuntu/MasterThesis/artificial_images"
finetune_folder = '/home/ubuntu/MasterThesis/yunus_rotated'
test_folder = '/home/ubuntu/MasterThesis/yunus_resized/test_adaptthresh'

# Process each image
def process_images_with_progress(folder):
    all_images = os.listdir(folder)
    total_images = len(all_images)
    processed_count = 0

    print(f"Total images to process: {total_images}\n")

    for img_name in all_images:
        img_path = os.path.join(folder, img_name)

        # loading
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to load {img_name}. Skipping.")
            continue

        # greyscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # adaptive thresholding
        thresholded_img = cv2.adaptiveThreshold(
            gray_img,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=11,  # Block size for local thresholding
            C=2           # Constant subtracted from the mean
        )

        cv2.imwrite(img_path, thresholded_img)

        # update progress
        processed_count += 1
        progress_percentage = (processed_count / total_images) * 100
        print(f"Progress: {processed_count}/{total_images} ({progress_percentage:.2f}%)", end="\r")

    print("\nProcessing complete!")

processed_folder = "/home/ubuntu/MasterThesis/processed_images"
os.makedirs(processed_folder, exist_ok=True)

def process_images_to_greyscale_and_threshold_separate(folder, processed_folder):
    for img_name in tqdm(os.listdir(folder), desc="Processing Images"):
        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to load {img_name}. Skipping.")
            continue

        # greyscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresholded_img = cv2.adaptiveThreshold(
            gray_img,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )

        processed_path = os.path.join(processed_folder, img_name)
        cv2.imwrite(processed_path, thresholded_img)

process_images_to_greyscale_and_threshold_separate(finetune_folder, '/home/ubuntu/MasterThesis/yunus_rotated_adaptthresh')

process_images_with_progress(output_folder)
process_images_with_progress(finetune_folder)
process_images_with_progress(test_folder)


data_like_yunus_path = '/home/ubuntu/MasterThesis/language_model_photos/data_like_yunus'
# Help - almost no storage capacity left!!! 
yunus_names = os.listdir(image_folder)
print(yunus_names)

save_to = '/home/ubuntu/MasterThesis/code/yunus_data/data_like_yunus.json'
with open(save_to, 'w') as json_file:
    json.dump(yunus_names, json_file, indent=4)



