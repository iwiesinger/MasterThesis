#region #### General Settings and Imports ####

#region Import packages
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from PIL import ImageFile
import pandas as pd
import numpy as np
from safetensors.torch import load_file
from transformers import AutoFeatureExtractor
from transformers import BertLMHeadModel, VisionEncoderDecoderModel, SwinModel, SwinConfig, BertConfig, VisionEncoderDecoderConfig
from safetensors.torch import safe_open
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import wandb
from transformers import default_data_collator
from torcheval.metrics import WordErrorRate
import json
from sklearn.metrics import classification_report
from transformers import TrainerCallback
from transformers import EarlyStoppingCallback


#endregion

#region General settings and directories - validation commented out and pretrained model commented out
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
root_dir_pretrain = "/home/ubuntu/MasterThesis/artificial_images"
root_dir_finetune = '/home/ubuntu/MasterThesis/yunus_rotated_adaptthresh'
root_dir_test =  "/home/ubuntu/MasterThesis/yunus_resized/test_adaptthresh"
#root_dir_val = '/home/ubuntu/MasterThesis/yunus_resized/validation'
pretrained_bert_path = '/home/ubuntu/MasterThesis/code/yunus_data/seventh_try_15epochs_10batch/checkpoint-420/'
output_dir = '/home/ubuntu/MasterThesis/code/yunus_data/final_finetuning_LANGUAGE'
safetensors_file = pretrained_bert_path + "model.safetensors"
train_data_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_art.json'
finetune_data_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_train_val_aug.json'
test_data_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_test_resized.json'
#val_data_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_val_big_aug.json'
vocab_path = '/home/ubuntu/MasterThesis/code/yunus_data/vocab.json'
inv_vocab_path = '/home/ubuntu/MasterThesis/code/yunus_data/inv_vocab.json'
#endregion


#region Import datasets and (inv) vocab from pretraining
with open(train_data_path, 'r') as f:
    df_pretrain = pd.DataFrame(json.load(f))

with open(finetune_data_path, 'r') as f:
    df_finetune = pd.DataFrame(json.load(f))

with open(test_data_path, 'r') as f:
    df_test = pd.DataFrame(json.load(f))

with open(vocab_path, 'r') as f:
    vocab = json.load(f)

with open(inv_vocab_path, 'r') as f:
    inv_vocab = json.load(f)
inv_vocab = {int(k): v for k, v in inv_vocab.items()}

#for key, value in list(vocab.items())[:5]:
#    print(f"{key}: {value}")
#vocab = {key: int(value) for key, value in vocab.items()}
#endregion


#region NEW Custom Class + Dataset Creation for treatment without vocab
# use tokenizer instead of vocab

#region NEW Custom Class + Dataset Creation
# Function for custom dataset creation
class TransliterationWithImageDataset(Dataset):
    def __init__(self, root_dir, df, vocab, feature_extractor, max_seq_len=512):
        self.root_dir = root_dir
        self.df = df
        self.vocab = vocab
        self.feature_extractor = feature_extractor
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['img_name']
        image_path = os.path.join(self.root_dir, img_name)

        try:
                image = Image.open(image_path).convert("RGB")  
        except OSError:
            raise RuntimeError(f"Image at {image_path} could not be opened.")

        # image
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values.squeeze()

        # input ids, attention mask
        input_ids = torch.tensor(self.df.iloc[idx]['input_ids'])
        attention_mask = torch.tensor(self.df.iloc[idx]['attention_mask'])

        # padding tokens shall be ignored
        labels = input_ids.clone()
        labels[input_ids == self.vocab['<PAD>']] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }



feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224")

# custom datasets and data loaders
pretrain_dataset_with_images = TransliterationWithImageDataset(root_dir=root_dir_pretrain, df=df_pretrain, vocab=vocab, feature_extractor=feature_extractor)
finetune_dataset_with_images = TransliterationWithImageDataset(root_dir=root_dir_finetune, df=df_finetune, vocab=vocab, feature_extractor=feature_extractor)
test_dataset_with_images = TransliterationWithImageDataset(root_dir=root_dir_test, df=df_test, vocab=vocab, feature_extractor=feature_extractor)

pretrain_loader_with_images = DataLoader(pretrain_dataset_with_images, batch_size=10, shuffle=True)
finetune_loader_with_images = DataLoader(finetune_dataset_with_images, batch_size=10, shuffle=True)
test_loader_with_images = DataLoader(test_dataset_with_images, batch_size=10)

#endregion

print('Number of training examples:', len(pretrain_dataset_with_images))  # Pretraining
print('Number of fine-tuning examples:', len(finetune_dataset_with_images))  # Fine-tuning 
print('Number of test examples:', len(test_dataset_with_images))  # Test



#region #### Model Setup ####
bert_config = BertConfig.from_pretrained(pretrained_bert_path)
bert_config.is_decoder = True 
print("Loaded vocab size from configuration:", bert_config.vocab_size)
bert_config.add_cross_attention = True  # Enable cross-attention for decoder
bert_config.vocab_size = 124 
print("Configured vocab size after manual change:", bert_config.vocab_size)
swin_config = SwinConfig()

# Initialize SwinBERT VisionEncoderDecoder model
config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(swin_config, bert_config)
model = VisionEncoderDecoderModel(config=config)

# Initialize the encoder and decoder separately
model.encoder = SwinModel(swin_config)
model.decoder = BertLMHeadModel.from_pretrained(pretrained_bert_path, config=bert_config)
#endregion#endregion

#region Model statistics and documentation
def  model_size(model):
  return sum(t.numel() for t in model.parameters())

start_size = f'START SIZE:\nSwin size: {model_size(model.encoder)/1000**2:.1f}M parameters\BERT size: {model_size(model.decoder)/1000**2:.1f}M parameters\nSwin+BERT size: {model_size(model)/1000**2:.1f}M parameters\n'
print(start_size)
#START SIZE:
#Swin size: 27.5M parameters\BERT size: 114.5M parameters
#Swin+BERT size: 142.0m parameters

# Write the strings
def build_text_files(data_list, dest_path):
    f = open(dest_path, 'w')
    f.write(data_list)

build_text_files(start_size, output_dir + '/start_size.txt')
#endregion

#region Vocabulary Matching
# Special Token IDs - according to TOKENIZER
model.config.pad_token_id = vocab['<PAD>']
model.config.decoder_start_token_id = vocab['<BOS>']  
model.config.eos_token_id = vocab['<EOS>'] 
model.config.unk_token_id = vocab['<UNK>'] 

# vocabulary size
model.config.vocab_size = len(vocab)  # Vocab Size == Number of unique tokens
model.decoder.resize_token_embeddings(len(vocab)) 


print(f"Model config:\nPad token ID: {model.config.pad_token_id}\nBOS token ID: {model.config.decoder_start_token_id}\nEOS token ID: {model.config.eos_token_id}\nUnknown token ID: {model.config.unk_token_id}\nVocab size: {model.decoder.config.vocab_size}")

#Model config:
#Pad token ID: 123
#BOS token ID: 120
#EOS token ID: 121
#Unknown token ID: 122
#Vocab size: 124

#endregion

#region Beam search parameters
model.config.early_stopping = False
model.config.max_length = 191 # covers 90% of all observations in length
#model.config.no_repeat_ngram_size = 100
model.config.length_penalty = 1.4
model.config.num_beams = 4
epochs = 10*1
batch_size = 10
#eval_steps = np.round(len(df_train) / batch_size * epochs / 20, 0)
logging_steps = np.round(len(df_pretrain) / batch_size * epochs / 10, 0)  
#endregion

#region .safetensors file
# Load pretrained BERT weights from safetensors file and move them to decoder
# Hintergrund: I got an error message when opening the safetensors file about the BERT prefixes

with safe_open(safetensors_file, framework="pt", device="cpu") as f:
    state_dict = {key: f.get_tensor(key) for key in f.keys()}

# Filter weights that match the SwinBERT decoder
filtered_state_dict = {key: value for key, value in state_dict.items() if "bert." in key}
# Load only matching weights
missing_keys, unexpected_keys = model.decoder.load_state_dict(filtered_state_dict, strict=False)
#print(f"Missing keys: {missing_keys}")
#print(f"Unexpected keys: {unexpected_keys}")

# Initialize missing layers (cross-attention)
model.decoder.apply(model.decoder._init_weights)  # Randomly initializes only the missing layers
# Load state dict into decoder 
#model.decoder.load_state_dict(torch.load(pretrained_bert_path + "model.safetensors"))

# Verify loaded weights by checking the model size
def model_size(model):
    return sum(t.numel() for t in model.parameters())

print(f'Start Size:\nSwin size: {model_size(model.encoder)/1000**2:.1f}M parameters\nBERT size: {model_size(model.decoder)/1000**2:.1f}M parameters\nSwin+BERT size: {model_size(model)/1000**2:.1f}M parameters')
#Start Size:
#Swin size: 27.5M parameters
#BERT size: 114.5M parameters
#Swin+BERT size: 142.0M parameters
# Stilll the same
#endregion


#region Evaluation Metric

import wandb


def decode_ids(ids, inv_vocab):
    """Decode a list of token IDs into a string using the inverse vocabulary and print debug information."""
    # ids must be iterable
    if isinstance(ids, int):
        ids = [ids]
    elif not isinstance(ids, (list, tuple)):
        raise ValueError(f"Expected `ids` to be a list or int, got {type(ids)}: {ids}")

    # decode
    decoded_tokens = []
    for token_id in ids:
        if token_id in inv_vocab:
            decoded_tokens.append(inv_vocab[token_id])
        else:
            print(f"Token ID {token_id} not found in `inv_vocab`.")

    # return
    return " ".join(decoded_tokens)

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # replace -100 with pad
    if isinstance(labels_ids, np.ndarray):
        labels_ids[labels_ids == -100] = vocab['<PAD>']
    elif isinstance(labels_ids, list):
        labels_ids = [
            np.where(np.array(seq) == -100, vocab['<PAD>'], seq).tolist()
            for seq in labels_ids
        ]
    else:
        raise ValueError("labels_ids must be a list or NumPy array.")

    # decode predictions and labels
    try:
        pred_str = [
            decode_ids(ids.tolist() if isinstance(ids, np.ndarray) else ids, inv_vocab)
            for ids in pred_ids
        ]
        label_str = [
            decode_ids(ids.tolist() if isinstance(ids, np.ndarray) else ids, inv_vocab)
            for ids in labels_ids
        ]

        # Ensure lengths match
        if len(pred_str) != len(label_str):
            raise ValueError(
                f"Mismatch in number of predictions ({len(pred_str)}) and labels ({len(label_str)})."
            )
    except Exception as e:
        return {"eval_error": f"Error during decoding: {str(e)}"}

    # Calculate Token Error Rate (TER)
    try:
        # initialize metric
        metric = WordErrorRate()
        metric.update(pred_str, label_str)
        ter = metric.compute()
    except Exception as e:
        print(f"Error during TER calculation: {e}")
        return {"eval_error": f"Error during TER calculation: {str(e)}"}

    return {"ter": ter.item()} # float


#endregion


#region #### Training ####
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.empty_cache() # memory <3
#torch.cuda.reset_peak_memory_stats()
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
model.to(device)

# Initialize wandb
wandb.init(project="large_data_pretraining", name="LANGUAGE_10epochs10batch_try2_different_pretraining")
wandb.config.update({"model/num_parameters": model.num_parameters()}, allow_val_change=True) 


#region Pretraining
pretraining_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,               # generation during evaluation
    eval_strategy="no",                       # no evaluation during pretraining
    num_train_epochs=10,                  
    per_device_train_batch_size=batch_size,  
    fp16=True,                                # mixed precision for faster training
    save_strategy="epoch",                   
    save_total_limit=1,                       # keep one checkpoint (last one)
    output_dir=output_dir + "/pretraining",   # saving pretraining
    logging_dir='./logs_pretraining',         # logging
    logging_steps=logging_steps,              
    report_to="wandb",                        # WANDB
)

pretraining_trainer = Seq2SeqTrainer(
    model=model,                              
    args=pretraining_args,                 
    train_dataset=pretrain_dataset_with_images,  
    data_collator=default_data_collator,     
)

pretraining_trainer.train()

# save pretrained model
pretrained_model_path = output_dir + "/pretrained_model_v2"
pretrained_model_path = '/home/ubuntu/MasterThesis/code/yunus_data/final_finetuning_LANGUAGE/pretrained_model_v2'
model.save_pretrained(pretrained_model_path)
wandb.finish()
#endregion

#region Finetuning

wandb.init(project="large_data_finetuning", name="finetune_LANGUAGE_try1_15epochs10batch_v2")
wandb.config.update({"model/num_parameters": model.num_parameters()}, allow_val_change=True)

# load pretrained model
model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_path)

# training arguments for finetuning
finetuning_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,               # generation during evaluation
    eval_strategy="no",                       # No evaluation 
    num_train_epochs=15,            
    per_device_train_batch_size=batch_size,   
    fp16=True,                                
    save_strategy="epoch",                    
    save_total_limit=1,                      # saving one
    output_dir=output_dir + "/fine_tuning",   
    logging_dir='./logs_finetuning',   
    logging_steps=logging_steps,             
    report_to="wandb",                      # wandb saving
)

# trainer for fine-tuning
finetuning_trainer = Seq2SeqTrainer(
    model=model,                              # Pretrained model
    args=finetuning_args,                     # Fine-tuning arguments
    train_dataset=finetune_dataset_with_images,  # second dataset: fine-tuning
    data_collator=default_data_collator,     
    compute_metrics=compute_metrics,          
)

finetuning_trainer.train()

fine_tuned_model_path = output_dir + "/fine_tuned_model_v2"
model.save_pretrained(fine_tuned_model_path)

test_results = finetuning_trainer.evaluate(test_dataset_with_images)

final_ter = test_results.get("eval_ter", None)
if final_ter is not None:
    wandb.log({"Test Token Error Rate (TER)": final_ter})
else:
    print("TER could not be calculated. Check the evaluation results:", test_results)


wandb.finish()
#endregion
