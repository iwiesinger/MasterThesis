
#region Import packages
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
#endregion

labels_path = '/home/ubuntu/'
model_path = '/home/ubuntu/'
output_dif = '/home/ubuntu/'

# Load labels 
language_df = pd.read_csv(labels_path + 'df_raw_nx.csv')
print(language_df.head())
random_seed = 43
language_df_shuffled = language_df.sample(frac=1, random_state = random_seed).reset_index(drop=True)
print(language_df_shuffled.head())

def train_test_split(df):
    train_split = int(0.8*len(df))
    df_train = df[:train_split]
    df_test= df[train_split:]
    return df_train, df_test

train_df, test_df = train_test_split(language_df_shuffled)
print(len(train_df)) #17562
print(len(test_df)) #4391

# Number of training and validation exmaples
num_train_ex = len(train_df)
num_val_ex = len(test_ex)

train_df = train_df.head(num_train_ex)
val_df = val_df.head(num_test_ex)

train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)

print(f'\nTrain data frame: \n{len(train_df)}\n\nTest data frame: \n{len(test_df)}\n')
print(f'\nTrain data frame: \n{train_df.head()}\n\nTest data frame: \n{test_df.head()}\n')


# Defining Synthetic Dataset Class

class SyntheticDataset(Dataset):
    def __init__(self, df, image_folder):
        self.df = df
        self.image_folder = image_folder
        #self.max_seq_length = max_seq_length

        # Filter out rows without corresponding image or text data
        self.df = self.df[self.df['_id'].apply(lambda x: os.path.exists(os.path.join(image_folder, f"{x}.jpg")))]
        print(f"Filtered dataset length: {len(self.df)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        id = self.df.iloc[idx]['_id']
        tok_text = self.df.iloc[idx]['tok_signs']

        #prepare image
        image_path = os.path.join(self.image_folder, f"{id}.jpg")
        image = Image.open(image_path).convert("RGB")

        






# Load architectures in the model
config_encoder = SwinConfig()
config_decoder = BertConfig()


# Group architectures and define model
config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
model = VisionEncoderDecoderModel(config=config)




