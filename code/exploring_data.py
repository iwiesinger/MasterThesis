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
id_list = list(id_unique)
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
import json
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

print(categories_coco[54:58]) #id 56 is an Unclear Sign


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


######### Database entries #########

#region MongoDB Setup
import pymongo
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from gridfs import GridFSBucket
mongo_uri = "mongodb://wiesinger:xtz4bbBHnyDKeYT4uqEW@badwcai-ebl01.srv.mwn.de:27017,badwcai-ebl02.srv.mwn.de:27018,badwcai-ebl03.srv.mwn.de:27019/ebl?replicaSet=rs-ebl&authSource=ebl&authMechanism=SCRAM-SHA-1&tls=true&tlsAllowInvalidCertificates=true"

try:
    # Create a MongoDB client
    client = MongoClient(mongo_uri)

    # The ping command is used to check if the connection to MongoDB is successful
    client.admin.command('ping')
    print("Connection to MongoDB successful!")

except ConnectionFailure as e:
    print(f"Connection failed: {e}")

db = client['ebl']

######################YEEEEEEES; THAT WORKED!!!!!!!!!!! ############################
#endregion

#region Retrieve Images

#region Checking capacity
import shutil
total, used, free = shutil.disk_usage("/")

print(f"Total: {total / (1024**3):.2f} GB")
print(f"Used: {used / (1024**3):.2f} GB")
print(f"Free: {free / (1024**3):.2f} GB")
#endregion

# Connect to the specific database

# Access the GridFS bucket for 'photos'
grid_fs_bucket = GridFSBucket(db, bucket_name="photos")

file_names_without_extension = id_list  
output_folder = 'language_model_photos'

# Loop through each filename, add ".jpg", and download the corresponding file
""" for file_name in file_names_without_extension:
    full_file_name = f"{file_name}.jpg"  # Add the ".jpg" extension
    output_file_path = os.path.join(output_folder, full_file_name)

    try:
        # Open a stream to the file in GridFS by name
        with grid_fs_bucket.open_download_stream_by_name(full_file_name) as grid_out:
            # Read the file's binary content
            file_data = grid_out.read()
            
            # Save file into folder
            with open(output_file_path, 'wb') as output_file:
                output_file.write(file_data)
                
        print(f"Downloaded {full_file_name} to {output_file_path}")
    
    except Exception as e:
        print(f"Error downloading {full_file_name}: {e}")
 """

import os
from PIL import Image

folder_path = 'language_model_photos'
print(len(os.listdir(folder_path)))
### 19354 images downloaded ###
#endregion

#region Explore Image characteristics

# Get a list of all jpg files in the folder
jpg_files = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]

# Iterate through each image and print its size (width and height in pixels)
for file_name in jpg_files:
    # Open the image file
    with Image.open(os.path.join(folder_path, file_name)) as img:
        # Get the size (width, height) of the image
        width, height = img.size
        print(f'Image: {file_name} | Width: {width} px | Height: {height} px')
# Images are not normalized

#endregion

#region Getting IDs to match to category names
signs_collection = db['signs']

signs_dict = {}

for sign in signs_collection.find():
    sign_id = sign.get('_id')
    lists = sign.get('lists', [])

    found_match = False

    for entry in lists:
        name = entry.get('name')
        number = entry.get('number')
    
        if name and 'ABZ' in name and isinstance(number, (int, str)):
            signs_dict[sign_id] = f"{name}{number}"
            print(f'Match found! Added {sign_id}: {name}{number} to sign_dict.')
            found_match = True
            break

    if not found_match:
        print(f'No match found for sign {sign_id}')

print(signs_dict)
print(len(signs_dict)) #827 signs
#endregion

#region Using this to convert COCO categories into ABZ codes
import pandas as pd
categories_coco = annotations_train_coco['categories']
coco_categories_df = pd.DataFrame(categories_coco)

def get_encoded_name(name):
    return signs_dict.get(name, "Unknown") 

coco_categories_df['encoded_name'] = coco_categories_df['name'].apply(get_encoded_name)
print(coco_categories_df)
unknown_entries = coco_categories_df[coco_categories_df['encoded_name']== 'Unknown']
print(unknown_entries) 
#endregion

#region Checking if all COCO categories come up in Language Model Dataset
# COCO categories stored in coco_categories_df['encoded_name]
# compare to unique_token_counts[unique_tok_counts{train_nx}]
uni_tok_train_nx = unique_token_counts['unique_tok_counts_train_nx']
print(uni_tok_train_nx)

matches = coco_categories_df['encoded_name'].isin(uni_tok_train_nx['token'])
print(matches)
if matches.all():
    print("All entries from 'encoded name' are present in 'unique_tok_count_train_nx'.")
else:
    missing_entries = coco_categories_df[~matches]
    print("The following 'encoded_name' entries are missing in 'unique_tok_counts_train_nx':")
    print(missing_entries)
#      id           name encoded_name
#49    49          |I.A|      ABZ142a
#56    56    UnclearSign      Unknown
#86    86        |U.GUD|       ABZ441
#93    93      |IGI.DIB|       ABZ455
#96    96          |U.U|       ABZ471
#103  103         |U.KA|       ABZ412
#108  108        |U.U.U|       ABZ472

# All of them are there in the dataset, just as an unsure sign -> if we increase the dataset, it would appear and be recognized.

#endregion
