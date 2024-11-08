
#region Import packages
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from safetensors.torch import load_file
#endregion

labels_path = '/home/ubuntu/MasterThesis/'
model_path = '/home/ubuntu/MasterThesis/'
output_dir = '/home/ubuntu/MasterThesis/'

#region Defining Custom class for handling both image and text data

class TransliterationWithImageDataset(Dataset):
    def __init__(self, root_dir, df, vocab, feature_extractor, max_seq_len=520):
        self.root_dir = root_dir
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.feature_extractor = feature_extractor
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image file and text data
        id = self.df['_id'][idx]
        image_path = f"{self.root_dir}{id}.jpg"
        image = Image.open(image_path).convert("RGB")
        
        # Process image
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values.squeeze()
        
        # Get input_ids and attention_mask from the dataframe
        input_ids = self.df['input_ids'][idx]
        attention_mask = self.df['attention_mask'][idx]
        
        # Convert to tensors
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        
        # Replace padding token IDs with -100 to ignore them in the loss
        labels = input_ids.clone()
        labels[input_ids == self.vocab['<PAD>']] = -100
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

#endregion

#region Creating custom class datasets
root_dir = '/home/ubuntu/MasterThesis/language_model_photos/'

# Load SWIN feature extractor
from transformers import AutoFeatureExtractor
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224")

# Create datasets
train_dataset_with_images = TransliterationWithImageDataset(df=df_train, root_dir=root_dir, feature_extractor=feature_extractor, vocab=vocab)
test_dataset_with_images = TransliterationWithImageDataset(df=df_test, root_dir=root_dir, feature_extractor=feature_extractor, vocab=vocab)

# Create data loaders
train_loader_with_images = DataLoader(train_dataset_with_images, batch_size=20, shuffle=True)
test_loader_with_images = DataLoader(test_dataset_with_images, batch_size=20)

print('Number of training examples:', len(train_dataset_with_images))
print('Number of test examples:', len(test_dataset_with_images))

# Verifying example from training dataset
encoding = train_dataset_with_images[0]
print(encoding)
for k,v in encoding.items():
    print(k,v.shape) #69 signs

img_ex = Image.open(train_dataset_with_images.root_dir + df_train['_id'][0] + ".jpg").convert("RGB")
img_ex.show()

# Decode the labels

labels = encoding['labels']
# Replace -100 with padding token ID to decode (or filter out -100)
decoded_tokens = [inv_vocab[id.item()] for id in labels if id.item() != -100]
decoded_text = " ".join(decoded_tokens)
print("Decoded text:", decoded_text)
print(len(decoded_text))
#endregion

#region loading in the pretrained BERT weights
from transformers import BertModel, VisionEncoderDecoderModel, SwinModel, SwinConfig, BertConfig, VisionEncoderDecoderConfig

pretrained_bert_path = '/home/ubuntu/MasterThesis/model_results_pretraining_train_test/checkpoint-11196/'

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

def  model_size(model):
  return sum(t.numel() for t in model.parameters())

start_size = f'START SIZE:\nSwin size: {model_size(model.encoder)/1000**2:.1f}M parameters\BERT size: {model_size(model.decoder)/1000**2:.1f}M parameters\nSwin+BERT size: {model_size(model)/1000**2:.1f}M parameters\n'
print(start_size)

# Write the strings
def build_text_files(data_list, dest_path):
    f = open(dest_path, 'w')
    f.write(data_list)

build_text_files(start_size, output_dir + '/start_size.txt')


#endregion

#region Vocabulary Matching
#from transformers import PreTrainedTokenizerFast

# Special Token IDs
model.config.pad_token_id = vocab['<PAD>']
model.config.decoder_start_token_id = vocab['<BOS>']  
model.config.eos_token_id = vocab['<EOS>'] 
model.config.unk_token_id = vocab['<UNK>'] 

# vocabulary size
model.config.vocab_size = len(vocab)  # Number of unique tokens
model.decoder.resize_token_embeddings(len(vocab))

print(f"Model config:\nPad token ID: {model.config.pad_token_id}\nBOS token ID: {model.config.decoder_start_token_id}\nEOS token ID: {model.config.eos_token_id}\nUnknown token ID: {model.config.unk_token_id}\nVocab size: {model.config.vocab_size}")
#endregion

#model.decoder.load_state_dict(torch.load(pretrained_bert_path + "model.safetensors"))
# Load the pretrained BERT weights from the safetensors file

pretrained_weights = load_file(pretrained_bert_path + "model.safetensors")
for key in pretrained_weights.keys():
    print(key)
# Remove "bert." prefix from the keys
adjusted_weights = {k.replace("bert.", ""): v for k, v in pretrained_weights.items()}
model.decoder.load_state_dict(pretrained_weights)

#region Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

epochs=60*1
batch_size=20
eval_steps = np.round(len(train_df)/batch_size*epochs/20,0)
logging_steps=eval_steps

# Training arguments
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",  
    num_train_epochs=epochs,  
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True,
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

