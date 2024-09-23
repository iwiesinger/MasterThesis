######### Overview & Preprocessing #########

#region Read and get overview over raw data
#JSON file path
import json
file_path = 'Akkadian.json'

import os
os.getcwd()


# Check if the file exists and is readable
if os.path.exists(file_path) and os.access(file_path, os.R_OK):
    try:
        # Open and load data
        with open(file_path, 'r') as file:
            raw_data = json.load(file)
            print(f"Data is readable")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
else:
    print("The file does not exist or is not readable.")

# Length of the list
list_length = len(raw_data)
print(f"The raw data has {list_length} items.")

# Types of elements
element_types = set(type(item) for item in raw_data)
print(f"The list contains items of types: {element_types}")

# convert into dataframe
import pandas as pd
df_raw = pd.DataFrame(raw_data)
print(df_raw.head())
print(df_raw.columns) # id, script, signs
print(len(df_raw)) #22054 observations
print(type(df_raw))
print(type(df_raw['script']))
print(df_raw['script'].head())
#endregion

#region Variable Overviews 
#region Period variable (in signs)
# Looking at period
df_raw['period'] = df_raw['script'].apply(lambda x: x['period'])

# Get the unique values of the 'period' column 
unique_periods = df_raw['period'].unique()
unique_periods_list = unique_periods.tolist()
print(unique_periods_list)
print(len(unique_periods_list))
#endregion

#region Period Counts
period_counts = df_raw['period'].value_counts()
print(period_counts)
#endregion

#region PeriodModifier variable (in signs)
df_raw['periodModifier'] = df_raw['script'].apply(lambda x: x['periodModifier'])
unique_period_modifier = df_raw['periodModifier'].unique()
unique_period_modifier = unique_period_modifier.tolist()
print(unique_period_modifier)

#endregion

#region Uncertain
df_raw['uncertain'] = df_raw['script'].apply(lambda x: x['uncertain'])
uncertain = df_raw['uncertain'].unique()
uncertain = uncertain.tolist()
print(uncertain)
#endregion

#region sortKey
df_raw['sortKey'] = df_raw['script'].apply(lambda x: x['sortKey'])
sortKey = df_raw['sortKey'].unique()
sortKey = sortKey.tolist()
print(sortKey)
print(len(sortKey))
#endregion

#region ID 
print(df_raw['_id'].head(5))
id_unique = df_raw['_id'].unique()
print(len(id_unique))
# ID is a unique identifier of each observation
#endregion
#endregion



######### Looking at annotations used for cuneiform-ocr project #########

#region icdar2015
import json
import pandas as pd
with open('textdet_test.json', 'r') as file: 
    annotations_icdar= json.load(file)

print(type(annotations_icdar))
print(annotations_icdar.keys()) 
for key, value in annotations_icdar.items():
    print(f'{key}: {len(value)}')
    print(f'{key}: {value[1:5]}')
    print(f'type: {type(annotations_icdar[value])}')
print(annotations_icdar['metainfo'].keys())
print(annotations_icdar['metainfo'].values().unique())

# Extract the metainfo dictionary
metainfo = annotations_icdar["metainfo"]
print(type(metainfo['category']))
print(metainfo['category']['id'])

# Get unique values for 'dataset_type' and 'task_name'
unique_dataset_type = set([metainfo["dataset_type"]])
unique_task_name = set([metainfo["task_name"]])

# unique 'id' and 'name' in the 'category' list
unique_ids = set()
unique_names = set()

for category in metainfo['category']:
    unique_ids.add(category['id'])
    unique_names.add(category['name'])

# Print the unique values
print("Unique values for dataset_type:", unique_dataset_type)
print("Unique values for task_name:", unique_task_name)
print("Unique values for category 'id':", unique_ids)
print("Unique values for category 'name':", unique_names)

# icdar2015 is a dictionary with keys "meta_info" and "data_list"
# "meta_info" is itself a dictionary with keys "dataset_type" (str), "task_name" (str) and "category" (list)
# "dataset_type" is a string with the value 'TextDataset' for all observations
# "task_name" is a string with the value 'textdet' for all observations
# "category" is a list of dictionaries with keys "id" and "name"
# "id" has value '0' for all observations
# "name" has value 'text' for all observations


#endregion

#region coco
with open('instances_train2017.json', 'r') as file: 
    annotations_train_coco = json.load(file)

print(type(annotations_train_coco))
print(annotations_train_coco.keys()) 
for key, value in annotations_train_coco.items():
    print(f'{key}: {len(value)}')
    print(f'{key}: {value[1]}')
    print(f'type: {type(value)}')

# Access the 'instances' list
images_coco_annot = annotations_train_coco['images']
annotations_coco = annotations_train_coco['annotations']
categories_coco = annotations_train_coco['categories']
print(type(images_coco_annot))

instances = [images_coco_annot, annotations_coco, categories_coco]

# Check if 'instances' is a list and if the elements are dictionaries, then get their keys
for i, instance in enumerate(instances):
    if isinstance(instance, list) and len(instance) > 0 and isinstance(instance[0], dict):
        print(f"Keys in instance {i} (first element): {instance[0].keys()}")
    else:
        print(f"Instance {i} is not a list of dictionaries")


# Run this code with ALL THE KEYS within annotations
unique_cat_coco = {}

# check if value is a dictionary
if isinstance(categories_coco, list) and len(categories_coco) > 0 and isinstance(categories_coco[0], dict):
    # Initialize sets for each key in the dictionaries
    for key in categories_coco[0].keys():
        unique_cat_coco[key] = set()

    # Iterate over each dictionary in the list
    for cat in categories_coco:
        # Collect unique values for each key
        for key, value in cat.items():
            # Check if the value is a list or dict
            if isinstance(value, (list, dict)):
                # Convert lists to tuples or process if needed
                value = tuple(value) if isinstance(value, list) else str(value)
            unique_cat_coco[key].add(value)

# Print the unique values and their counts 
print("Unique values for 'categories':")
for key, values in unique_cat_coco.items():
    print(f"  {key}: {len(values)} unique values")

# check values
for i, category in enumerate(categories_coco):
    print(f"Processing category {i}: {category}")  
    if "id" in category:
        print(f"id: {category['id']}")
    else:
        print(f"'id' not found in category {i}")
# coco-recognition training annotations is a dictionary which has the keys "images", "annotations" and "categories"
#   images is a list of length 554 with (100 for validation) with list values that are dictionaries 
#       (keys: "id", "file_name", "height", "width")
#       "id" has 554 unique values and (0-553, sorted like that)
#       "file_name" has 554 unique values
#       
#   annotations is a list of length 46595 with list values that are dictionaries 
#       (keys: "image_id", "id", "category_id", "bbox", "area", "segmentation", "scrowd")
#       "image_id" has 554 unique values (): enumerates from 1 to 553 (images)
#       "id" has 46595 unique values, enumerates until 46594 
#       "category_id" has 120 unique values, that will be the different signs
#       "segmentation" has one value which is always empty
#       "iscrowd" has one value which is always zero
#   categories is a list of length 141 (120 for coco-recognition) with list values that are dictionaries
#       (keys: "id", "name")
#       "id" has 120 unique values from 0 to 119
#       "name" has 120 unique values (categories)
#endregion

#region output_new
with open('output_new.json', 'r') as file:
    annotations_new = json.load(file)
print(type(annotations_new))
print(annotations_new[0].keys())
print(len(annotations_new))

# Initialize sets to store unique values for each key
unique_ocredSigns = set()
unique_filenames = set()

# Iterate over the list to collect unique values
for entry in annotations_new:
    # Add values for each key to their respective sets
    if "ocredSigns" in entry:
        unique_ocredSigns.add(entry["ocredSigns"])
    if "filename" in entry:
        unique_filenames.add(entry["filename"])

# Print the unique values and their counts
print(f"Unique values for 'ocredSigns': {len(unique_ocredSigns)} unique values")
print(f"Unique values for 'filename': {len(unique_filenames)} unique values")

# the new output is a list of 72779 instances. Each list contains three dictionaries: 
#   "ocredSign" has 59828 unique values (=lists with the exact same ABZ codes)
#   "file_name" has 72423 unique values (72423 unique files)
#   "ocredSignsCoordinates"

#endregion



