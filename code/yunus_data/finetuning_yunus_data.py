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

#region General settings and directories
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
root_dir_train = "/home/ubuntu/MasterThesis/yunus_big_art_random_augment"
root_dir_test =  "/home/ubuntu/MasterThesis/yunus_resized/test"
root_dir_val = '/home/ubuntu/MasterThesis/yunus_resized/validation'
pretrained_bert_path = '/home/ubuntu/MasterThesis/code/yunus_data/pretraining_after_better_reorder/checkpoint-840/'
output_dir = '/home/ubuntu/MasterThesis/code/yunus_data/finetune_big_art_random_augment'
safetensors_file = pretrained_bert_path + "model.safetensors"
train_data_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_train_big_art_aug.json'
test_data_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_test_resized.json'
val_data_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_val_big_aug.json'
vocab_path = '/home/ubuntu/MasterThesis/code/yunus_data/vocab.json'
inv_vocab_path = '/home/ubuntu/MasterThesis/code/yunus_data/inv_vocab.json'
#endregion


#region Import datasets and (inv) vocab from pretraining
with open(train_data_path, 'r') as f:
    df_train = pd.DataFrame(json.load(f))

with open(test_data_path, 'r') as f:
    df_test = pd.DataFrame(json.load(f))

with open(val_data_path, 'r') as f:
    df_val = pd.DataFrame(json.load(f))

with open(vocab_path, 'r') as f:
    vocab = json.load(f)

with open(inv_vocab_path, 'r') as f:
    inv_vocab = json.load(f)
inv_vocab = {int(k): v for k, v in inv_vocab.items()}

for key, value in list(vocab.items())[:5]:
    print(f"{key}: {value}")
vocab = {key: int(value) for key, value in vocab.items()}
#endregion


print(len(df_train))

print(df_train.head())

# Compute the lengths of lists in the 'tok_signs' column
df_train['tok_signs_length'] = df_train['tok_signs'].apply(len)
length_stats = df_train['tok_signs_length'].describe()
print(length_stats)

quantiles_10_percent = df_train['tok_signs_length'].quantile([i / 10 for i in range(11)])
quantiles_10_percent

#endregion

print(df_train.head())

#region #### Data Prep ####
#region OLD Custom Class + Dataset Creation - back when the data in big_data was used
'''class TransliterationWithImageDataset(Dataset):
    def __init__(self, root_dir, df, vocab, feature_extractor, max_seq_len=512, max_pixels=178956970):
        self.root_dir = root_dir
        self.vocab = vocab
        self.feature_extractor = feature_extractor
        self.max_seq_len = max_seq_len

        # Filter out rows where the corresponding image file doesn't exist, done once here
        self.df = df[df['_id'].apply(lambda x: os.path.exists(f"{self.root_dir}{x}.jpg"))].reset_index(drop=True)

        # Cache for resized images
        self.image_cache = {}

        # Threshold dimensions for resizing
        self.resize_threshold = (1000, 1000)
        self.min_size = (32, 32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        while idx < len(self.df):
            # Get image file and text data
            id = self.df['_id'][idx]
            image_path = f"{self.root_dir}{id}.jpg"
            
            # Check if resized img is already in cache
            if id in self.image_cache:
                pixel_values, original_shape, resized_shape = self.image_cache[id]
            else:
                # Load image
                try:
                    image = Image.open(image_path).convert("RGB")
                except OSError:
                    print(f"Skipping corrupted image: {image_path}")
                    idx + 1
                    continue
                original_shape = image.size + (3,)  # Original width, height, channels

                # Skip images smaller than the minimum size
                if original_shape[0] < self.min_size[0] or original_shape[1] < self.min_size[1]:  # New: Skip small images
                    idx += 1  # Move to the next index
                    continue  # Skip to the next iteration

                # Check dimensions against threshold
                if original_shape[0] >= self.resize_threshold[0] and original_shape[1] >= self.resize_threshold[1]:
                    # Resize to approximately 50% of original pixel count if larger than threshold
                    scaling_factor = 0.7
                    new_size = (int(original_shape[0] * scaling_factor), int(original_shape[1] * scaling_factor))
                    resized_image = image.resize(new_size)
                    resized_shape = resized_image.size + (3,)
                else:
                    # Use original image without resizing
                    resized_image = image
                    resized_shape = original_shape

                # Process image through feature extractor
                pixel_values = self.feature_extractor(resized_image, return_tensors="pt").pixel_values.squeeze()

                # Cache processed data
                self.image_cache[id] = (pixel_values, original_shape, resized_shape)

            # input_ids, attention_masks
            input_ids = torch.tensor(self.df['input_ids'][idx])
            attention_mask = torch.tensor(self.df['attention_mask'][idx])

            # Replace padding token IDs with -100 to ignore them in loss
            labels = input_ids.clone()
            labels[input_ids == self.vocab['<PAD>']] = -100

            return {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'original_shape': original_shape,
                'resized_shape': resized_shape
            }'''
#endregion

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

        # Process image
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values.squeeze()

        # input_idds, attention_mask
        input_ids = torch.tensor(self.df.iloc[idx]['input_ids'])
        attention_mask = torch.tensor(self.df.iloc[idx]['attention_mask'])

        # Replace padding token IDs with -100 to ignore during loss calculation
        labels = input_ids.clone()
        labels[input_ids == self.vocab['<PAD>']] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }



# feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224")

#region Dataset Creation
train_dataset_with_images = TransliterationWithImageDataset(root_dir=root_dir_train, df=df_train, vocab=vocab, feature_extractor=feature_extractor)
test_dataset_with_images = TransliterationWithImageDataset(root_dir=root_dir_test, df=df_test, vocab=vocab, feature_extractor=feature_extractor)
val_dataset_with_images = TransliterationWithImageDataset(root_dir=root_dir_val, df=df_val, vocab=vocab, feature_extractor=feature_extractor)
#endregion


# Create data loaders
train_loader_with_images = DataLoader(train_dataset_with_images, batch_size=10, shuffle=True)
test_loader_with_images = DataLoader(test_dataset_with_images, batch_size=10)
val_loader_with_images = DataLoader(val_dataset_with_images, batch_size=10, shuffle=True)

print('Number of training examples:', len(train_dataset_with_images)) # 554 images
print('Number of validation examples:', len(val_dataset_with_images)) # 100 images
print('Number of test examples:', len(test_dataset_with_images)) # 2,886 images

#endregion


#region ProgressPrintCallback and Early Stopping Callback
class ProgressPrintCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current_step = state.global_step
        current_epoch = state.epoch
        ter = metrics.get("ter", "N/A")  # Get TER from metrics

        # Log TER to wandb if it's a valid metric
        if ter != "N/A":
            wandb.log({"Token Error Rate (TER)": ter, "Step": current_step, "Epoch": current_epoch})

        print(f"Step {current_step}, Epoch {current_epoch}, TER: {ter}")


# early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=15,  
    early_stopping_threshold=0.001  
)

#endregion



#region #### Model Setup ####
# Model configuration Swin and BERT
bert_config = BertConfig.from_pretrained(pretrained_bert_path)
bert_config.is_decoder = True 
print("Loaded vocab size from configuration:", bert_config.vocab_size)
bert_config.add_cross_attention = True  # Enable cross-attention for decoder
bert_config.vocab_size = 124 
print("Configured vocab size after manual change:", bert_config.vocab_size)
swin_config = SwinConfig()

# Initialization
config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(swin_config, bert_config)
model = VisionEncoderDecoderModel(config=config)

# separately
model.encoder = SwinModel(swin_config)
model.decoder = BertLMHeadModel.from_pretrained(pretrained_bert_path, config=bert_config)
#endregion

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
# Special Token IDs
model.config.pad_token_id = vocab['<PAD>']
model.config.decoder_start_token_id = vocab['<BOS>']  
model.config.eos_token_id = vocab['<EOS>'] 
model.config.unk_token_id = vocab['<UNK>'] 

# vocabulary size
model.config.vocab_size = len(vocab)  # Vocab Size == Number of unique tokens
model.decoder.resize_token_embeddings(len(vocab)) 

print(f"Model config:\nPad token ID: {model.config.pad_token_id}\nBOS token ID: {model.config.decoder_start_token_id}\nEOS token ID: {model.config.eos_token_id}\nUnknown token ID: {model.config.unk_token_id}\nVocab size: {model.config.vocab_size}")
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
epochs = 40*1
batch_size = 10
#eval_steps = np.round(len(df_train) / batch_size * epochs / 20, 0)
logging_steps = np.round(len(df_train) / batch_size * epochs / 20, 0)  
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
print(f"Missing keys: {missing_keys}")
print(f"Unexpected keys: {unexpected_keys}")

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
    # 'ids' needs to always be iterable (e.g., convert a single int to a list)
    if isinstance(ids, int):
        ids = [ids]
    elif not isinstance(ids, (list, tuple)):
        raise ValueError(f"Expected `ids` to be a list or int, got {type(ids)}: {ids}")

    # checking 
    print(f"Raw Token IDs: {ids}")

    # Decoding
    decoded_tokens = []
    for token_id in ids:
        if token_id in inv_vocab:
            decoded_tokens.append(inv_vocab[token_id])
        else:
            print(f"Token ID {token_id} not found in `inv_vocab`.")

    # checking decoded output
    print(f"Decoded Tokens: {decoded_tokens}")

    return " ".join(decoded_tokens)

'''
for idx, ids in enumerate(df_train['input_ids'][-5:]):
    print(f"Row {idx + 1} Input IDs: {ids}")
    decoded = decode_ids(ids, inv_vocab)
    print(f"Row {idx + 1} Decoded Output: {decoded}")'''
# Looks about right.

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Replace -100 with <PAD> in labels
    if isinstance(labels_ids, np.ndarray):
        labels_ids[labels_ids == -100] = vocab['<PAD>']  # Handle numpy arrays directly
    elif isinstance(labels_ids, list):
        labels_ids = [np.where(np.array(seq) == -100, vocab['<PAD>'], seq).tolist() for seq in labels_ids]
    else:
        raise ValueError("labels_ids must be a list or NumPy array.")
    # Decode predictions and labels
    pred_str = [decode_ids(ids.tolist() if isinstance(ids, np.ndarray) else ids, inv_vocab) for ids in pred_ids]
    label_str = [decode_ids(ids.tolist() if isinstance(ids, np.ndarray) else ids, inv_vocab) for ids in labels_ids]

    # TER as WER
    try:
        metric = WordErrorRate()
        ter = metric(pred_str, label_str).item()
    except Exception as e:
        return {"error": f"Error during TER calculation: {str(e)}"}

    # Log TER to wandb
    # wandb.log({"Token Error Rate (TER)": ter})

    return {"ter": ter}



#region Testing evaluation metric
test_ids = [0, 120, 121, 123] 
decoded_str = decode_ids(test_ids, inv_vocab)
#print(decoded_str)
# Function itself works!

#endregion

#region verify alignment
def verify_alignment(pred, inv_vocab, num_samples=5):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Ensure label_ids has the correct structure
    if not all(isinstance(seq, (list, np.ndarray)) for seq in labels_ids):
        raise ValueError("label_ids must be a list of sequences (list of lists). Found:", labels_ids)

    # Replace -100 with <PAD> and convert arrays to lists
    labels_ids = [list(np.where(np.array(seq) == -100, vocab['<PAD>'], seq)) for seq in labels_ids]
    print("Processed label_ids:", labels_ids)

    # Decode predictions and labels
    pred_str = [decode_ids(ids, inv_vocab) for ids in pred_ids]
    label_str = [decode_ids(seq, inv_vocab) for seq in labels_ids]

    # Compare samples
    for i in range(min(num_samples, len(label_str))):
        # Skip sequences that decode to only <PAD>
        if label_str[i] == "<PAD>" and pred_str[i] == "<PAD>":
            print(f"Sample {i + 1}: Skipped (only padding)")
            continue

        print(f"Sample {i + 1}:")
        print(f"  Ground Truth: {label_str[i]}")
        print(f"  Predicted: {pred_str[i]}")
        print("-" * 40)


#endregion



#endregion

#region #### Training ####
#region Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
model.to(device)

# Initialize wandb
wandb.init(project="master_thesis_finetuning", name="finetune_big_art_random_augment")

# Update the model/num_parameters key with allow_val_change=True
wandb.config.update({"model/num_parameters": model.num_parameters()}, allow_val_change=True) 

#region Training arguments with validation dataset

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,              
    eval_strategy="epoch",            
    num_train_epochs=epochs,               
    per_device_train_batch_size=batch_size, 
    per_device_eval_batch_size=batch_size,   
    fp16=True,                                #  mixed precision for faster training
    save_strategy="epoch",              
    save_total_limit=2,                       # Keep 2
    load_best_model_at_end=True,              
    metric_for_best_model="eval_ter",         # validation metric
    greater_is_better=False,                  # less is more 
    output_dir=output_dir,                   
    logging_dir='./logs',                     
    logging_steps=logging_steps,             
    report_to="wandb",                        
)

# Load the datasets and define the trainer

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_with_images,  # Training data
    eval_dataset=val_dataset_with_images,    # Validation data
    data_collator=default_data_collator,     
    compute_metrics=compute_metrics,         
    callbacks=[ProgressPrintCallback(), early_stopping_callback],  
)

#endregion
#endregion

#region Training Arguments without validation dataset
'''
# Training arguments
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,         # Enable generation during evaluation
    evaluation_strategy="no",          # No evaluation during training
    num_train_epochs=epochs,           # Set the number of epochs
    per_device_train_batch_size=batch_size,  # Training batch size
    per_device_eval_batch_size=batch_size,   # Evaluation batch size
    fp16=True,                         # Use mixed precision for faster training
    save_strategy="epoch",             # Save model at the end of each epoch
    output_dir=output_dir,             # Output directory for saving models
    logging_dir='./logs',              # Directory for logs
    logging_steps=logging_steps,       # Log every few steps
    report_to="wandb",                 # Log training progress to wandb
    save_total_limit=1,                # Only keep the last saved model
)

# Load the datasets and define the trainer

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_with_images,  # Training dataset
    eval_dataset=test_dataset_with_images,   # validation dataset (used as eval)
    data_collator=default_data_collator,     # Data collator for batching
    compute_metrics=compute_metrics,         # Evaluate TER during testing
    callbacks=[ProgressPrintCallback()],     # Progress callback
)'''
#endregion

#region Training & Evaluation
# Start training with wandb logging enabled
trainer.train()

# Run final evaluation
final_results = trainer.evaluate(test_dataset_with_images)


# logging
final_ter = final_results.get("eval_ter", None)  
if final_ter is not None:
    wandb.log({"Final Token Error Rate (TER)": final_ter})
else:
    print("TER could not be calculated. Check the evaluation results:", final_results)


# Finish the wandb run
wandb.finish()
#endregion


#endregion



#region Loading trained model and test it using test dataset
'''from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM
from datasets import Dataset
import wandb

# Define paths
checkpoint_dir = "/home/ubuntu/MasterThesis/finetuning_output/checkpoint-27020/"

# Load the trained model
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)

# Assuming your custom tokenizer was saved and loaded manually
tokenizer = CustomTokenizer(vocab=vocab)

# Load the test dataset
# Assuming `test_dataset_with_images` is preprocessed and tokenized already
# Example: test_dataset_with_images = Dataset.from_dict({"input_ids": ..., "labels": ..., "attention_mask": ...})

# Recreate the same training arguments (adjust `eval_dataset` only)
eval_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    per_device_eval_batch_size=8,  
    fp16=True, 
    output_dir="./eval_logs", 
    logging_dir='./eval_logs',
    report_to="wandb",  
    metric_for_best_model="ter",
    greater_is_better=False
)

# Recreate the trainer with only `eval_dataset`
trainer = Seq2SeqTrainer(
    model=model,
    args=eval_args,
    eval_dataset=test_dataset_with_images, 
    tokenizer=tokenizer,  
    compute_metrics=compute_metrics, 
    data_collator=default_data_collator 
)

# evaluate
test_results = trainer.evaluate()
print("Test Evaluation Results:", test_results)

# Logging
wandb.init(project="master-thesis-evaluation", name="test-evaluation")
final_ter = test_results.get("eval_ter", "N/A")
wandb.log({"Test Token Error Rate (TER)": final_ter})
wandb.finish()'''

#endregion
