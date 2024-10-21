
#region Import packages
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
#endregion

labels_path = '/home/ubuntu/MasterThesis/'
model_path = '/home/ubuntu/MasterThesis/'
output_dif = '/home/ubuntu/MasterThesis/'

#region Defining Custom class for handling both image and text data

class TransliterationWithImageDataset(Dataset):
    def __init__(self, df, root_dir, feature_extractor, max_seq_len=520):
        self.df = df
        self.root_dir = root_dir  # images directory
        self.feature_extractor = feature_extractor # SWIN
        self.max_seq_len = max_seq_len # unnecessary?

        # Convert input_ids and attention masks to tensors
        self.input_ids = torch.tensor(df['input_ids'].tolist())
        self.attention_mask = torch.tensor(df['attention_mask'].tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Text data: input_ids and attention mask
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]

        # Image data
        file_name = self.df['_id'][idx] + ".jpg"  # as _id maps to image file name
        image_path = self.root_dir + file_name
        image = Image.open(image_path).convert("RGB")

        # Extract pixel values from the image
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values.squeeze(),  # Removing extra batch dimension
            'labels': input_ids  # Assuming you're using input_ids as labels (e.g., for masked language modeling)
        }

#endregion

#region Creating custom class datasets
root_dir = 'language_model_photos/'

# Load the SWIN feature extractor
from transformers import AutoFeatureExtractor
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224")

# Create datasets
train_dataset_with_images = TransliterationWithImageDataset(df_train_nx, root_dir, feature_extractor)
val_dataset_with_images = TransliterationWithImageDataset(df_val_nx, root_dir, feature_extractor)
test_dataset_with_images = TransliterationWithImageDataset(df_test_nx, root_dir, feature_extractor)

# Create data loaders
train_loader_with_images = DataLoader(train_dataset_with_images, batch_size=24, shuffle=True)
val_loader_with_images = DataLoader(val_dataset_with_images, batch_size=24, shuffle=True)
test_loader_with_images = DataLoader(test_dataset_with_images, batch_size=24)
#endregion

#region loading in the pretrained BERT weights
from transformers import BertModel, VisionEncoderDecoderModel, SwinModel, SwinConfig, BertConfig, VisionEncoderDecoderConfig

pretrained_bert_path = 'results_2noNEWLINE/checkpoint-2564/'

# BERT configuration and model
bert_config = BertConfig.from_pretrained(pretrained_bert_path)
bert_model = BertModel.from_pretrained(pretrained_bert_path)

# Fision encoder (Swin)
swin_config = SwinConfig()
swin_model = SwinModel(swin_config)

# Combination into VisionEncoderDecoderModel
config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(swin_config, bert_config)
model = VisionEncoderDecoderModel(config=config)

# Decoder is pretrained BERT model
model.decoder = bert_model
#endregion

#region Vocabulary Matching
from transformers import PreTrainedTokenizerFast

# Special Token IDs
model.config.pad_token_id = vocab_nx['<PAD>']
model.config.decoder_start_token_id = vocab_nx['<BOS>']  
model.config.eos_token_id = vocab_nx['<EOS>'] 
model.config.unk_token_id = vocab_nx['<UNK>'] 

# vocabulary size
model.config.vocab_size = len(vocab_nx)  # Number of unique tokens
model.decoder.resize_token_embeddings(len(vocab_nx))

print(f"Model config:\nPad token ID: {model.config.pad_token_id}\nBOS token ID: {model.config.decoder_start_token_id}\nEOS token ID: {model.config.eos_token_id}\nVocab size: {model.config.vocab_size}")
#endregion

model.decoder.load_state_dict(torch.load(pretrained_bert_path + "model.safetensors"))

#region Training setup

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# Training arguments
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",  
    num_train_epochs=epochs,  
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_steps=500,  
    output_dir=output_dir,  
    logging_steps=logging_steps,  
    eval_steps=eval_steps,  
)

# Load the datasets and define the trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset_with_images,
    eval_dataset=val_dataset_with_images,
    data_collator=default_data_collator
)


#endregion


#region OLD First tries
language_df = pd.read_csv(labels_path + 'data/df_raw_nx.csv')
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

 #endregion       






# Load architectures in the model
config_encoder = SwinConfig()
config_decoder = BertConfig()


# Group architectures and define model
config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
model = VisionEncoderDecoderModel(config=config)




