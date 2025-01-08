########## Preparing texts that should be copied with bounding boxes of other images ##########

#region imports
import pandas as pd
import json

#endregion

df_train_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_train_big_aug.json'
vocab_path = '/home/ubuntu/MasterThesis/code/yunus_data/vocab.json'

with open(df_train_path, 'r') as f:
    df_train = pd.DataFrame(json.load(f))

print(df_train.columns)

# How many columns have leq 25 tokens? (because 25*40 = 1000)
df_train['tok_len'] = df_train['tok_signs'].apply(len)
toks_leq25 = df_train[df_train['tok_len']<=25]
toks_leq500 = df_train[df_train['tok_len']<=500]
print(len(toks_leq25)) # 369
print(len(toks_leq500)) # 1314

# Is the whole vocabulary covered by these 149 observations?
with open(vocab_path, 'r') as f:
    vocab = set(json.load(f))

all_tokens = set()
toks_leq25['tok_signs'].apply(lambda x: all_tokens.update(x))
print(len(toks_leq25))
print(all_tokens)
all_tokens500 = set()
toks_leq500['tok_signs'].apply(lambda x: all_tokens500.update(x))

missing_tokens500 = vocab - all_tokens500
coverage_percentage500 = (1 - len(missing_tokens500)/len(vocab))
print(f'Total vocabulary: {len(vocab)}, \nTokens in Dataset: {len(all_tokens500)}, \nCoverage percentage: {coverage_percentage500}, \nMissing tokens: {len(missing_tokens500)}')
# 2 missing tokens, 98% coverage
missing_tokens = vocab - all_tokens
coverage_percentage = (1 - len(missing_tokens) / len(vocab))
print(f'Total vocabulary: {len(vocab)}, \nTokens in Dataset: {len(all_tokens)}, \nCoverage percentage: {coverage_percentage}, \nMissing tokens: {len(missing_tokens)}')
# 6 tokens missing, 95% coverage


# Function to resize and pad image to 40x40 while preserving aspect ratio
def resize_and_pad(image_path, target_size):
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    delta_w = target_size - img.size[0]
    delta_h = target_size - img.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    padded_img = ImageOps.expand(img, padding, fill="white")
    return padded_img
import pandas as pd
from PIL import Image, ImageOps
import os
import random

# Configuration
input_folder = "/home/ubuntu/MasterThesis/train_set/"  
output_folder = "/home/ubuntu/MasterThesis/artificial_images"  
os.makedirs(output_folder, exist_ok=True)
canvas_size = 1024
sign_size = 40
letters_per_row = canvas_size // sign_size  # Maximum letters per row (25)
num_variations = 3  # Number of variations per row

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


# Function to generate synthetic images and create a new dataset
# Function to generate synthetic images with centered signs and a black background
def generate_artificial_images_and_dataset(toks_leq25):
    df_art = pd.DataFrame()  
    for index, row in toks_leq25.iterrows():
        sequence = row['abz'] 
        base_img_name = row['img_name'].rsplit('.jpg', 1)[0] 

        for variation in range(num_variations):
            # Create a blank canvas with a black background
            canvas = Image.new("RGB", (canvas_size, canvas_size), "black")

            # Calculate the total height and width of the sequence
            num_rows = (len(sequence) + letters_per_row - 1) // letters_per_row  # Rows needed
            total_height = num_rows * sign_size
            total_width = min(len(sequence), letters_per_row) * sign_size

            # Centering offsets for the sequence
            y_start = (canvas_size - total_height) // 2
            x_start = (canvas_size - total_width) // 2

            # Start placing tokens
            for idx, token in enumerate(sequence):
                row_idx = idx // letters_per_row
                col_idx = idx % letters_per_row
                x_offset = x_start + col_idx * sign_size
                y_offset = y_start + row_idx * sign_size

                # Load and resize the token image
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

                # Place the resized and padded image on the canvas
                canvas.paste(sign_img, (x_offset, y_offset))

            # Save the generated image
            new_img_name = f"{base_img_name}_art{variation + 1}.jpg"
            canvas.save(os.path.join(output_folder, new_img_name))

            # Add the new row to df_art
            new_row = row.copy()
            new_row['img_name'] = new_img_name
            df_art = pd.concat([df_art, pd.DataFrame([new_row])], ignore_index=True)

    return df_art



df_art = generate_artificial_images_and_dataset(toks_leq500)

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

count_files_in_folder('/home/ubuntu/MasterThesis/yunus_big_art_random_augment')