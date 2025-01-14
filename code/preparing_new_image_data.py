########## Preparing Text Data to be used for pretraining ##########
import os
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrainerCallback
import math
from transformers import EarlyStoppingCallback


image_folder = '/home/ubuntu/MasterThesis/language_model_photos/data_like_yunus'

img_names = [
    os.path.splitext(f)[0] for f in os.listdir(image_folder) 
    if os.path.isfile(os.path.join(image_folder, f)) and f.endswith(".jpg")
]

def count_files_in_folder(folder_path):
    length = len(os.listdir(folder_path))
    return length

files_in_img_folder = count_files_in_folder(image_folder) # 4366
print(f'{files_in_img_folder} images are in the bigger image folder')

img_names_df = pd.DataFrame(img_names, columns=["img_name"])

big_language_data_path = '/home/ubuntu/MasterThesis/code/big_language_data.json'
with open(big_language_data_path, 'r') as f:
    big_language_data = pd.DataFrame(json.load(f)) # 21 952 rows
print(len(big_language_data))

# Filter rows where _id matches image names
filt_big_language_data = big_language_data[big_language_data["_id"].isin(img_names)]
print(len(filt_big_language_data)) # 4366

big_language_data['_id'] = big_language_data['_id'].str.strip()
image_names_strip = [name.strip() for name in img_names]

matches = big_language_data["_id"].isin(image_names_strip)
matching_count = matches.sum()  # Count how many matches exist
print(matching_count) # All images have matching text files! 

matched_data = big_language_data.loc[matches, ["_id", "signs", 'tok_signs']]
print(matched_data.head())
# some observations seem to have many X signs! Let's see if all of them have enough good signs

def tokenize_signs(signs):
    signs = signs.replace('\n', ' <NEWLINE> ')  
    tokens = signs.split() 
    tokens = ['<BOS>'] + [token for token in tokens if token not in '<NEWLINE>'] + ['<EOS>']  
    return tokens

matched_data['tok_signs'] = matched_data['signs'].apply(tokenize_signs)
print(len(matched_data))
classes = ['ABZ13', 'ABZ579', 'ABZ480', 'ABZ70', 'ABZ597', 'ABZ342', 'ABZ461', 'ABZ381', 'ABZ61', 'ABZ1', 'ABZ142', 'ABZ318', 'ABZ231', 'ABZ75', 'ABZ449', 'ABZ533', 'ABZ354', 'ABZ139', 'ABZ545', 'ABZ536', 'ABZ330', 'ABZ308', 'ABZ86', 'ABZ328', 'ABZ214', 'ABZ73', 'ABZ15', 'ABZ295', 'ABZ296', 'ABZ68', 'ABZ55', 'ABZ69', 'ABZ537', 'ABZ371', 'ABZ5', 'ABZ151', 'ABZ411', 'ABZ457', 'ABZ335', 'ABZ366', 'ABZ324', 'ABZ396', 'ABZ206', 'ABZ99', 'ABZ84', 'ABZ353', 'ABZ532', 'ABZ58', 'ABZ384', 'ABZ376', 'ABZ59', 'ABZ334', 'ABZ74', 'ABZ383', 'ABZ589', 'ABZ144', 'ABZ586', 'ABZ7', 'ABZ97', 'ABZ211', 'ABZ399', 'ABZ52', 'ABZ145', 'ABZ343', 'ABZ367', 'ABZ212', 'ABZ78', 'ABZ85', 'ABZ319', 'ABZ207', 'ABZ115', 'ABZ465', 'ABZ570', 'ABZ322', 'ABZ331', 'ABZ38', 'ABZ427', 'ABZ279', 'ABZ112', 'ABZ79', 'ABZ80', 'ABZ60', 'ABZ535', 'ABZ142a', 'ABZ314', 'ABZ232', 'ABZ554', 'ABZ312', 'ABZ172', 'ABZ128', 'ABZ6', 'ABZ595', 'ABZ230', 'ABZ167', 'ABZ12', 'ABZ306', 'ABZ331e+152i', 'ABZ339', 'ABZ134', 'ABZ575', 'ABZ401', 'ABZ313', 'ABZ472', 'ABZ441', 'ABZ62', 'ABZ111', 'ABZ468', 'ABZ148', 'ABZ397', 'ABZ104', 'ABZ147', 'ABZ455', 'ABZ471', 'ABZ412', 'ABZ2', 'ABZ440', 'ABZ101', 'ABZ538', 'ABZ72', 'ABZ298', 'ABZ143', 'ABZ437', 'ABZ393', 'ABZ483', 'ABZ94', 'ABZ559', 'ABZ565', 'ABZ87', 'ABZ138', 'ABZ50', 'ABZ191', 'ABZ152', 'ABZ124', 'ABZ205', 'ABZ398', 'ABZ9', 'ABZ126', 'ABZ164', 'ABZ195', 'ABZ307', 'ABZ598a']

def mark_unknown_tokens(tokens, vocab):
    # Check if the token is in the vocabulary or one of the special tokens
    return [token if token in vocab or token in ['<BOS>', '<EOS>'] else '<UNK>' for token in tokens]

matched_data['tok_signs'] = matched_data['tok_signs'].apply(lambda tokens: mark_unknown_tokens(tokens, classes))
matched_data['signs_len'] = matched_data['tok_signs'].apply(len)
matched_data['X_percent'] = matched_data['tok_signs'].apply(lambda tokens: tokens.count('<UNK>') / len(tokens) * 100 if len(tokens) > 0 else 0)

print(matched_data[['signs_len', 'X_percent']])

matched_less10 = matched_data[matched_data['X_percent'] < 10] # 1590
matched_less15 = matched_data[matched_data['X_percent'] < 15] # 2486
matched_less20 = matched_data[matched_data['X_percent'] < 20] # 3077

print(len(matched_less20))
print(len(matched_less15))
print(len(matched_less10))
# I will choose matched_less15.
print(matched_less15.head())
matched_less15['img_name'] = matched_less15['_id'] + '.jpg'

matched_less15.to_json('/home/ubuntu/MasterThesis/code/yunus_data/matches_largerdata.json', orient = 'records', indent = 4)
df_new = matched_less15
df_new['img_name'] = df_new['_id']+ '.jpg'

#region Import other training data
train_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_train_resized.json'
test_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_val_resized.json'

with open(train_path, 'r') as f:
    df_train = pd.DataFrame(json.load(f))

with open(test_path, 'r') as f:
    df_test = pd.DataFrame(json.load(f))

print(df_train.columns)
print(matched_less15.columns)
#endregion


#region NEW: Open Vocabulary and Inv Vocabulary
# Load vocab.json
with open('/home/ubuntu/MasterThesis/code/yunus_data/vocab.json', "r") as vocab_file:
    vocab = json.load(vocab_file)

# Load inv_vocab.json
with open('/home/ubuntu/MasterThesis/code/yunus_data/inv_vocab.json', "r") as inv_vocab_file:
    inv_vocab = json.load(inv_vocab_file)
#endregion


#region Convert tokenized data to input IDs and attention masks
def tokens_to_ids(tokens, vocab, max_len=512):
    token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    token_ids = token_ids[:max_len]  
    attention_mask = [1] * len(token_ids)
    padding_length = max_len - len(token_ids)
    token_ids = token_ids + [vocab['<PAD>']] * padding_length
    attention_mask = attention_mask + [0] * padding_length
    return token_ids, attention_mask

# Create input IDs and attention masks for both datasets
df_train['input_ids'], df_train['attention_mask'] = zip(*df_train['tok_signs'].apply(lambda x: tokens_to_ids(x, vocab)))
df_train['input_ids'] = df_train['input_ids'].apply(list)  # Ensure input_ids is a list
df_train['attention_mask'] = df_train['attention_mask'].apply(list)  # Ensure attention_mask is a list

#df_test['input_ids'], df_test['attention_mask'] = zip(*df_test['tok_signs'].apply(lambda x: tokens_to_ids(x, vocab)))
df_test['input_ids'], df_test['attention_mask'] = zip(*df_test['tok_signs'].apply(lambda x: tokens_to_ids(x, vocab)))
df_test['input_ids'] = df_test['input_ids'].apply(list)  # Ensure input_ids is a list
df_test['attention_mask'] = df_test['attention_mask'].apply(list) 

df_new['input_ids'], df_new['attention_mask'] = zip(*df_new['tok_signs'].apply(lambda x: tokens_to_ids(x, vocab)))
df_new['input_ids'] = df_new['input_ids'].apply(list)  # Ensure input_ids is a list
df_new['attention_mask'] = df_new['attention_mask'].apply(list) 


def tokens_to_labels(token_ids, pad_token_id=123):
    return [token_id if token_id != pad_token_id else -100 for token_id in token_ids]

df_new["labels"] = df_new["input_ids"].apply(lambda ids: tokens_to_labels(ids, pad_token_id=vocab["<PAD>"]))
df_train["labels"] = df_train["input_ids"].apply(lambda ids: tokens_to_labels(ids, pad_token_id=vocab["<PAD>"]))
df_test["labels"] = df_test["input_ids"].apply(lambda ids: tokens_to_labels(ids, pad_token_id=vocab["<PAD>"]))

df_new.to_json('/home/ubuntu/MasterThesis/code/yunus_data/df_new.json', orient = 'records', indent=4)

columns_to_keep = ['img_name', 'tok_signs', 'attention_mask', 'input_ids', 'labels']

# Ensure both datasets only have the selected columns
df_new_trimmed = df_new[columns_to_keep]
df_train_trimmed = df_train[columns_to_keep]

# Concatenate the two datasets along rows
combined_data = pd.concat([df_train_trimmed, df_new_trimmed], ignore_index=True)
print(combined_data.head())

df_combined_shuffled = combined_data.sample(frac=1, random_state=1235).reset_index(drop=True)
    
# Updated train-validation-test split function - USE WHEN IN NEEED OF VALIDATION DATA
'''def train_val_split(df):
    train_split = int(0.8 * len(df))

    df_train = df[:train_split]
    df_val = df[train_split:]

    return df_train, df_val

df_train_new, df_val_new = train_val_split(df_combined_shuffled)
'''

#region Create PyTorch Datasets and DataLoaders
class TransliterationDataset(Dataset):
    def __init__(self, df):
        self.input_ids = torch.tensor(df['input_ids'].tolist())
        self.attention_mask = torch.tensor(df['attention_mask'].tolist())
        self.labels = torch.tensor(df['labels'].tolist())
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]  
        }
    


# Create datasets without X
train_dataset = TransliterationDataset(df_combined_shuffled)
test_dataset = TransliterationDataset(df_test)
#val_dataset = TransliterationDataset(df_val_new)

# Create the dataloaders without X
train_loader = DataLoader(train_dataset, batch_size=15, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=15)
#val_loader = DataLoader(val_dataset, batch_size = 15, shuffle = True)



#region Perplexity Callback for wandb Logging

class PerplexityLoggingCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        # Extract the metrics logged during evaluation
        metrics = kwargs.get("metrics", {})
        eval_loss = metrics.get("eval_loss")

        if eval_loss is not None:
            perplexity = math.exp(eval_loss)
            wandb.log({"epoch": state.epoch, "perplexity": perplexity})
            print(f"Epoch {state.epoch}: Perplexity = {perplexity}")
#endregion

# earlystopping
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=5  
)


#region NEW BertLMHead
import wandb
from transformers import BertLMHeadModel, Trainer, TrainingArguments, BertConfig
from transformers.integrations import WandbCallback
import torch
import os
from transformers import EarlyStoppingCallback

torch.cuda.memory_summary(device=None, abbreviated=False)
torch.cuda.empty_cache()
torch.cuda.is_available()

# gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

wandb.init(project="master_thesis", name="additional_language_data_noval")

# load models
config = BertConfig.from_pretrained('bert-base-uncased')
config.is_decoder = True
config.vocab_size = len(vocab)
model = BertLMHeadModel.from_pretrained('bert-base-uncased', config=config, ignore_mismatched_sizes=True)
model.resize_token_embeddings(len(vocab))

output_dir = '/home/ubuntu/MasterThesis/code/yunus_data/pretraining_new_language_data_noval/'


training_args = TrainingArguments(
    output_dir=output_dir,             
    num_train_epochs=15,              
    per_device_train_batch_size=15,    
    warmup_steps=400,                 
    weight_decay=0.001,               
    logging_dir='./logs',             
    logging_steps=500,                 
    save_strategy="epoch",            
    save_total_limit=1,                
    report_to="wandb",                
    fp16=True                         
)

# Initialize the Trainer using train and test datasets (without validation dataset)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,       # Train on the train dataset
    callbacks=[
        WandbCallback(), 
        PerplexityLoggingCallback()]
) 



trainer.train()

test_result = trainer.evaluate(eval_dataset=test_dataset)
print("Test Loss: ", test_result['eval_loss'])

import math
test_perplexity = math.exp(test_result['eval_loss'])
print("Test Perplexity: ", test_perplexity)

trainer.save_model(output_dir)
print(f"Model saved to {output_dir}")

# End wandb logging
wandb.finish()

########## Seeking corresponding images and adding them to the image dataframes

import os
import shutil
from PIL import Image, ImageOps
import os

def filter_images_by_dataset(image_folder, df_new, output_folder, img_column='img_name'):
    """
    Filter images from a folder based on a dataset and copy them to a new folder.

    Args:
        image_folder (str): Path to the folder containing all images.
        df_new (pd.DataFrame): DataFrame containing the image names to filter.
        output_folder (str): Path to the new folder where the filtered images will be stored.
        img_column (str): Column name in the DataFrame containing the image names.

    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)

    image_names = df_new[img_column].tolist()

    for img_name in image_names:
        src_path = os.path.join(image_folder, img_name)
        dest_path = os.path.join(output_folder, img_name)

        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
        else:
            print(f"Warning: {img_name} not found in {image_folder}")

    print(f"Filtered images have been copied to {output_folder}.")
print(df_new.columns)

image_folder = "/home/ubuntu/MasterThesis/language_model_photos/data_like_yunus/"
output_folder = "/home/ubuntu/MasterThesis/language_model_photos/new_photos"
filter_images_by_dataset(image_folder, df_new, output_folder)

def count_files_in_folder(folder_path):
    length = len(os.listdir(folder_path))
    return length

count = count_files_in_folder('/home/ubuntu/MasterThesis/language_model_photos/new_photos')
print(len(df_new)) # 2486
print(count)

#region Resize images in the folder
import os
from shutil import copy2

def overwrite_images(input_folder, replacement_folder):
    # Ensure the folders exist
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist.")
        return
    if not os.path.exists(replacement_folder):
        print(f"Replacement folder '{replacement_folder}' does not exist.")
        return

    # List files in the replacement folder
    replacement_files = {file for file in os.listdir(replacement_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))}

    for filename in os.listdir(input_folder):
        # Skip non-image files
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue

        input_file_path = os.path.join(input_folder, filename)

        # Check if the image exists in the replacement folder
        if filename in replacement_files:
            replacement_file_path = os.path.join(replacement_folder, filename)
            try:
                copy2(replacement_file_path, input_file_path)
                print(f"Overwritten: {filename}")
            except Exception as e:
                print(f"Failed to overwrite {filename}: {e}")
        else:
            print(f"No replacement found for: {filename}")

input_folder = "/home/ubuntu/MasterThesis/language_model_photos/new_photos"
replacement_folder = "/home/ubuntu/MasterThesis/language_model_photos/data_like_yunus"

overwrite_images(input_folder, replacement_folder)

from PIL import Image, ImageOps
import os

def resize_and_pad_images(folder_path, target_size=(1024, 1024)):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Skip non-image files
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue

        with Image.open(file_path) as img:
            # Check if the image already matches the target size
            if img.size == target_size:
                print(f"Skipping {filename}, already 1024x1024.")
                continue

            # Resize the image while maintaining the aspect ratio
            img.thumbnail(target_size, Image.Resampling.LANCZOS)

            # Create a new 1024x1024 image with a black background
            padded_img = Image.new("RGB", target_size, (0, 0, 0))
            # Center the resized image on the canvas
            offset = ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2)
            padded_img.paste(img, offset)

            # Overwrite the original image
            padded_img.save(file_path)
            print(f"Processed and resized: {filename}")

# Folder path containing the images
input_folder = "/home/ubuntu/MasterThesis/language_model_photos/new_photos"
resize_and_pad_images(input_folder)



#endregion



# Copy pasted the 1662 augmented yunus training images into the folder

df_aug_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_train_aug_newrotation.json'

with open(df_aug_path, 'r') as f:
    df_aug = pd.DataFrame(json.load(f))

print(df_aug.columns)
print(df_new.columns)

columns_to_keep = ['img_name', 'tok_signs', 'attention_mask', 'input_ids', 'labels']


df_aug_trimmed = df_aug[columns_to_keep]
df_new_trimmed = df_new[columns_to_keep]

df_aug_new = pd.concat([df_aug_trimmed, df_new_trimmed], ignore_index=True)


df_aug_new.to_json('/home/ubuntu/MasterThesis/code/yunus_data/df_aug_new.json', orient = 'records', indent = 4)










