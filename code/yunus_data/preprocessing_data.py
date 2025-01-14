#region Imports
import json
import pandas as pd
#endregion


akkadian_path = '/home/ubuntu/MasterThesis/data/Akkadian.json'
coco_recog_train_path = '/home/ubuntu/instances_train2017.json'
coco_recog_val_path = '/home/ubuntu/instances_val2017_orig.json'
picture_path = '/home/ubuntu/MasterThesis/language_model_photos/'
yunus_pictures_train = '/home/ubuntu/MasterThesis/yunus_photos/train2017/'
yunus_pictures_val = '/home/ubuntu/MasterThesis/yunus_photos/val2017/'

#region reading in data
with open(akkadian_path, 'r') as f:
    akkadian = pd.DataFrame(json.load(f))

with open(coco_recog_train_path, 'r') as f:
    coco_recog_train = json.load(f)

with open(coco_recog_val_path, 'r') as f:
    coco_recog_val = json.load(f)
#endregion

#region Creating different datasets: a whole one, and validation and test data
# Add 'image_name' to annotations in coco_recog_train
train_image_id_to_name = {image['id']: image['file_name'] for image in coco_recog_train['images']}
for annotation in coco_recog_train['annotations']:
    annotation['image_name'] = train_image_id_to_name.get(annotation['image_id'], None)

# Add 'image_name' to coco_recog_val
val_image_id_to_name = {image['id']: image['file_name'] for image in coco_recog_val['images']}
for annotation in coco_recog_val['annotations']:
    annotation['image_name'] = val_image_id_to_name.get(annotation['image_id'], None)

'''# Print a list of image_names from the annotations of coco_recog_val
val_image_names = [annotation['image_name'] for annotation in coco_recog_val['annotations'] if 'image_name' in annotation]
print("Image names in coco_recog_val annotations:")
for name in val_image_names:
    print(name)'''


#region Split coco_recog_val into coco_recog_val (first 50 images) and coco_recog_test (last 50 images)

# split images
val_images = coco_recog_val['images'][:50]
test_images = coco_recog_val['images'][50:]

# get corresponding annotation
val_image_ids = {image['id'] for image in val_images}
test_image_ids = {image['id'] for image in test_images}
print(f"Validation image ids: {val_image_ids},\n Test image ids: {test_image_ids}")

val_annotations = [annotation for annotation in coco_recog_val['annotations'] if annotation['image_id'] in val_image_ids]
test_annotations = [annotation for annotation in coco_recog_val['annotations'] if annotation['image_id'] in test_image_ids]

# keep both categories
categories = coco_recog_val['categories']

coco_recog_val_split = {
    'images': val_images,
    'annotations': val_annotations,
    'categories': categories
}

coco_recog_test = {
    'images': test_images,
    'annotations': test_annotations,
    'categories': categories
}

'''# Save the datasets to JSON files
val_output_path = '/home/ubuntu/coco_recog_val.json'
test_output_path = '/home/ubuntu/coco_recog_test.json'

with open(val_output_path, 'w') as val_file:
    json.dump(coco_recog_val_split, val_file, indent=4)

with open(test_output_path, 'w') as test_file:
    json.dump(coco_recog_test, test_file, indent=4)

print(f"Validation dataset saved to {val_output_path}")
print(f"Test dataset saved to {test_output_path}")
'''
#endregion

#region Combining dataset
coco_recog = {
    'images': sorted(coco_recog_train['images'] + coco_recog_val['images'], key=lambda x: x['file_name']),
    'annotations': coco_recog_train['annotations'] + coco_recog_val['annotations'],
    'categories': coco_recog_train['categories']}

print("File names in coco_recog:")
for image in coco_recog['images']:
    print(image['file_name'])

print(len(coco_recog['annotations']))
coco_recog['annotations'][0]
coco_recog['annotations'][1]

# inspect
print("First 100 entries of coco_recog['annotations']:")
for annotation in coco_recog['annotations'][:400]:
    print(annotation)



# add abz notation
classes_path = '/home/ubuntu/MasterThesis/code/yunus_data/classes.txt'

# using class file
with open(classes_path, 'r') as file:
    class_list = eval(file.read()) 

# Add the 'abz' key to each category in 'categories'
for category in coco_recog['categories']:
    category_id = category['id']
    if 0 <= category_id < len(class_list):
        category['abz'] = class_list[category_id]


# verify
print("Updated categories with 'abz':")
for category in coco_recog['categories']:
    print(category)


#endregion

#region Extract file names and check how many sub-images there are
unique_suffixes = set()
for image in coco_recog['images']:
    file_name = image['file_name']
    if file_name.endswith('.jpg'):
        suffix = file_name[-6:-4]  
        unique_suffixes.add(suffix)

print("Unique last two characters before '.jpg':")
for suffix in sorted(unique_suffixes):
    print(suffix)
#endregion


#endregion

#region Create dictionary of how many annotations there are per sub-picture and whole picture
# Create a dictionary to store the counts of annotations per image_name and order by base_name
def count_annotations_by_base_name(coco_recog):
    # initialize empty
    annotation_counts = {}

    # iterate through coco_reg
    for annotation in coco_recog['annotations']:
        # get img name
        image_name = annotation['image_name']

        # count
        if image_name in annotation_counts:
            annotation_counts[image_name] += 1
        else:
            annotation_counts[image_name] = 1

    # order by base_name and group annotations by base_name
    grouped_annotations = {}

    for image_name, count in annotation_counts.items():
        # Extract  base name from the image_name
        if '-' in image_name:
            base_name = image_name.split('-')[0]
        else:
            base_name = image_name.replace('.jpg', '')

        # Add to grouped dictionary
        if base_name in grouped_annotations:
            grouped_annotations[base_name].append({"image_name": image_name, "count": count})
        else:
            grouped_annotations[base_name] = [{"image_name": image_name, "count": count}]

    return grouped_annotations

# new dictionary outside the function to sum counts by base name
def calculate_base_img_annotation_counts(grouped_annotations):
    base_img_annotation_counts = {}

    for base_name, sub_images in grouped_annotations.items():
        base_img_annotation_counts[base_name] = sum(item["count"] for item in sub_images)

    return base_img_annotation_counts

grouped_annotations = count_annotations_by_base_name(coco_recog)
base_img_annotation_counts = calculate_base_img_annotation_counts(grouped_annotations)
print(grouped_annotations)
print(base_img_annotation_counts)
#endregion

#region Prepare Akkadian dataset: Tokenization
def tokenize_signs_column(df, column_name="signs"):
    df["tok_signs"] = df[column_name].str.replace(r"\\n", r" \n ", regex=True).str.split()
    return df

akkadian["signs_with_spaces"] = akkadian["signs"].str.replace(r"\n", r" \n ", regex=True)

akkadian["signs_with_newline"] = akkadian["signs_with_spaces"].str.replace("\n", "NEWLINE")
# Tokenize the column with NEWLINE tokens
akkadian["tok_signs"] = akkadian["signs_with_newline"].str.split()

akkadian["tok_len"] = akkadian["tok_signs"].apply(len)
akkadian["newline_freq"] = akkadian["tok_signs"].apply(lambda x: x.count("NEWLINE"))
akkadian['normal_tok'] = akkadian['tok_len'] - akkadian['newline_freq']

print(akkadian[['tok_len', 'newline_freq', 'normal_tok']].head())
#endregion

#region Check if whole picture tokens are equal to expected whole picture tokens (depending on dataset)

# Compare normal_tok in akkadian with base_img_annotation_counts
mismatches = []

for base_name, count in base_img_annotation_counts.items():
    # Get the normal_tok value for the matching _id
    normal_tok_value = akkadian.loc[akkadian["_id"] == base_name, "normal_tok"].values

    # Check if there's a mismatch
    if len(normal_tok_value) == 0 or normal_tok_value[0] != count:
        mismatches.append({
            "base_name": base_name,
            "normal_tok": normal_tok_value[0] if len(normal_tok_value) > 0 else None,
            "expected": count
        })

# Print mismatches, if any
if mismatches:
    print("Mismatches found:")
    for mismatch in mismatches:
        print(f"Base Name: {mismatch['base_name']}, Normal_Tok: {mismatch['normal_tok']}, Expected: {mismatch['expected']}")
else:
    print("All counts match!")

#endregion

# I wanted to check if the transliterations could be created based on the big_dataset, as this one may contain the whole images 
# (that yunus_data contained the sub-images from)
# I applied a heuristic to see if it could work
# Unfortunately, this was a dead end. :()


#region Checking if all images exist in the folder
import os

# Directory containing the images
image_dir = '/home/ubuntu/MasterThesis/yunus_photos/val2017/'

# Function to check if all images exist
def check_images_exist(dataset, dataset_name):
    missing_files = []
    
    for image in dataset['images']:
        file_name = image['file_name']
        file_path = os.path.join(image_dir, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)

    if missing_files:
        print(f"{len(missing_files)} missing files in {dataset_name}:")
        for missing in missing_files:
            print(missing)
    else:
        print(f"All files in {dataset_name} exist.")

# Check for coco_recog_train
check_images_exist(coco_recog_train, "coco_recog_train")

# Check for coco_recog_val
check_images_exist(coco_recog_val, "coco_recog_val")

for image in val_images:
    print(image['file_name'])


# Unfortunately, most images don't exist in the originally downloaded big image folder, which might be due to partitioning of images. 
# In the paper it is mentioned that images that contain multiple tablet pictures are cut down into more parts.
# This means, I have to upload the exact images.
# Therefore, I have uploaded the images and redone the process, now all image files are available.
#endregion


