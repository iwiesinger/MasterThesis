#region #### General Settings and Imports ####

#region Import packages
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from safetensors.torch import load_file
from transformers import AutoFeatureExtractor
from transformers import BertLMHeadModel, VisionEncoderDecoderModel, SwinModel, SwinConfig, BertConfig, VisionEncoderDecoderConfig
from safetensors.torch import safe_open
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import wandb
from transformers import default_data_collator
from torchmetrics import WordErrorRate
import json
from sklearn.metrics import classification_report
from transformers import TrainerCallback


#endregion

#region General settings and directories
Image.MAX_IMAGE_PIXELS = None
root_dir = '/home/ubuntu/MasterThesis/language_model_photos/'
pretrained_bert_path = '/home/ubuntu/MasterThesis/results_pretrain_2024112_noval/'
output_dir = '/home/ubuntu/MasterThesis/finetuning_output/'
safetensors_file = pretrained_bert_path + "model.safetensors"
train_data_path = '/home/ubuntu/MasterThesis/data/df_train.json'
val_data_path = '/home/ubuntu/MasterThesis/data/df_val.json'
test_data_path = '/home/ubuntu/MasterThesis/data/df_test.json'
vocab_path = '/home/ubuntu/MasterThesis/data/vocab.json'
inv_vocab_path = '/home/ubuntu/MasterThesis/data/inv_vocab.json'
#endregion

#region Import datasets and (inv) vocab from pretraining
with open(train_data_path, 'r') as f:
    df_train = pd.DataFrame(json.load(f))

with open(val_data_path, 'r') as f:
    df_val = pd.DataFrame(json.load(f))

with open(test_data_path, 'r') as f:
    df_test = pd.DataFrame(json.load(f))

with open(vocab_path, 'r') as f:
    vocab = json.load(f)

with open(inv_vocab_path, 'r') as f:
    inv_vocab = json.load(f)
inv_vocab = {int(k): v for k, v in inv_vocab.items()}

for key, value in list(vocab.items())[:5]:
    print(f"{key}: {value}")
vocab = {key: int(value) for key, value in vocab.items()}
#endregion

#endregion


#region #### Data Prep ####
#region Custom Class + Dataset Creation
class TransliterationWithImageDataset(Dataset):
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image file and text data
        id = self.df['_id'][idx]
        image_path = f"{self.root_dir}{id}.jpg"
        
        # Check if resized img is already in cache
        if id in self.image_cache:
            pixel_values, original_shape, resized_shape = self.image_cache[id]
        else:
            # Load image
            image = Image.open(image_path).convert("RGB")
            original_shape = image.size + (3,)  # Original width, height, channels

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

            # Cache the processed image data
            self.image_cache[id] = (pixel_values, original_shape, resized_shape)

        # Get input_ids and attention_mask from the DataFrame
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
        }
#endregion

#region Creating image+text dataframes and dataloaders
def test_image_resizing(dataset, num_samples=10):
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        original_shape = sample['original_shape']
        resized_shape = sample['resized_shape']
        
        # Print original and resized shapes 
        print(f"Sample {i+1}")
        print(f"  Original shape: {original_shape}")
        print(f"  Resized shape: {resized_shape}")
        
        # Compare sizes
        original_pixel_count = original_shape[0] * original_shape[1]
        resized_pixel_count = resized_shape[0] * resized_shape[1]
        print(f"  Pixel count reduced to {resized_pixel_count / original_pixel_count * 100:.2f}% of original size\n")

# feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224")

# Filter DataFrame for images that exist - before creating datasets
df_train_filtered = df_train[df_train['_id'].apply(lambda x: os.path.exists(f"{root_dir}{x}.jpg"))].reset_index(drop=True)
df_val_filtered = df_val[df_val['_id'].apply(lambda x: os.path.exists(f"{root_dir}{x}.jpg"))].reset_index(drop=True)
df_test_filtered = df_test[df_test['_id'].apply(lambda x: os.path.exists(f"{root_dir}{x}.jpg"))].reset_index(drop=True)

# Create datasets with FILTERED DataFrame
train_dataset_with_images = TransliterationWithImageDataset(df=df_train_filtered, root_dir=root_dir, feature_extractor=feature_extractor, vocab=vocab)
val_dataset_with_images = TransliterationWithImageDataset(df=df_val_filtered, root_dir=root_dir, feature_extractor=feature_extractor, vocab=vocab)
test_dataset_with_images = TransliterationWithImageDataset(df=df_test_filtered, root_dir=root_dir, feature_extractor=feature_extractor, vocab=vocab)


# Test image resizing function
test_image_resizing(train_dataset_with_images)

# Create data loaders
train_loader_with_images = DataLoader(train_dataset_with_images, batch_size=10, shuffle=True)
val_loader_with_images = DataLoader(val_dataset_with_images, batch_size=10, shuffle=True)
test_loader_with_images = DataLoader(test_dataset_with_images, batch_size=10)

print('Number of training examples:', len(train_dataset_with_images)) # 16,374 images
print('Number of validation examples:', len(val_dataset_with_images)) # 2,868 images
print('Number of test examples:', len(test_dataset_with_images)) # 2,886 images

#endregion

#region test input_ids, attention_masks and labels are transferrred correctly
def test_input_ids_attention_mask_labels(dataset, original_df, num_samples=5):
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        input_ids = sample['input_ids']
        attention_mask = sample['attention_mask']
        labels = sample['labels']

        print(f"Sample {i+1}")
        print(f"  Input IDs: {input_ids.tolist()}")
        print(f"  Attention Mask: {attention_mask.tolist()}")
        print(f"Labels: {labels.tolist()}")

        # Verify that input_ids and attention_mask match the original DataFrame
        original_input_ids = original_df['input_ids'][i]
        original_attention_mask = original_df['attention_mask'][i]
        original_label = original_df['labels'][i]

        print(f"  Matches Original Input IDs: {torch.equal(input_ids, torch.tensor(original_input_ids))}")
        print(f"  Matches Original Attention Mask: {torch.equal(attention_mask, torch.tensor(original_attention_mask))}\n")
        print(f"  Matches Original Labels: {torch.equal(labels, torch.tensor(original_label))}\n")
        

test_input_ids_attention_mask_labels(train_dataset_with_images, df_train_filtered)


#endregion



#region ProgressPrintCallback
class ProgressPrintCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current_step = state.global_step
        current_epoch = state.epoch
        ter = metrics.get("ter", "N/A")  # Get TER from metrics

        # Log TER to wandb if it's a valid metric
        if ter != "N/A":
            wandb.log({"Token Error Rate (TER)": ter, "Step": current_step, "Epoch": current_epoch})

        print(f"Step {current_step}, Epoch {current_epoch}, TER: {ter}")

#endregion


#endregion 

#region #### Model Setup ####
#region Model setup & initialization: Swin & BERT
# Load the model configurations for Swin & BERT
bert_config = BertConfig.from_pretrained(pretrained_bert_path)
print("Loaded vocab size from configuration:", bert_config.vocab_size)
bert_config.add_cross_attention = True  # Enable cross-attention for decoder
bert_config.vocab_size = 6171 
print("Configured vocab size after manual change:", bert_config.vocab_size)

swin_config = SwinConfig()

# Initialize SwinBERT VisionEncoderDecoder model
config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(swin_config, bert_config)
model = VisionEncoderDecoderModel(config=config)

# Initialize the encoder and decoder separately
model.encoder = SwinModel(swin_config)
model.decoder = BertLMHeadModel.from_pretrained(pretrained_bert_path, config=bert_config)
#endregion

#region Model statistics and documentation
def  model_size(model):
  return sum(t.numel() for t in model.parameters())

start_size = f'START SIZE:\nSwin size: {model_size(model.encoder)/1000**2:.1f}M parameters\BERT size: {model_size(model.decoder)/1000**2:.1f}M parameters\nSwin+BERT size: {model_size(model)/1000**2:.1f}M parameters\n'
print(start_size)
#START SIZE:
#Swin size: 27.5M parameters\BERT size: 119.2M parameters
#Swin+BERT size: 146.7M parameters

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
#Pad token ID: 0
#BOS token ID: 2
#EOS token ID: 34
#Unknown token ID: 1
#Vocab size: 6171

for key, value in list(vocab.items())[:3]:
    print(f"{key}: {value}")

#endregion

#region Beam search parameters
model.config.early_stopping = True
model.config.max_length = 134 # covers 80% of all observations in length
#model.config.no_repeat_ngram_size = 100
model.config.length_penalty = 2.0
model.config.num_beams = 4

epochs = 20*1
batch_size = 10
eval_steps = np.round(len(df_train) / batch_size * epochs / 50, 0)
logging_steps = eval_steps
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
#BERT size: 119.2M parameters
#Swin+BERT size: 146.7M parameters
# -> Model size SMALLER - maybe because of smaller vocab?
#endregion


#region Evaluation Metric

import wandb


def decode_ids(ids, inv_vocab):
    """Decode a list of token IDs into a string using the inverse vocabulary and print debug information."""
    # Ensure 'ids' is always iterable (e.g., convert a single int to a list)
    if isinstance(ids, int):
        ids = [ids]
    elif not isinstance(ids, (list, tuple)):
        raise ValueError(f"Expected `ids` to be a list or int, got {type(ids)}: {ids}")

    # Print the raw input
    print(f"Raw Token IDs: {ids}")

    # Decode each token ID
    decoded_tokens = []
    for token_id in ids:
        if token_id in inv_vocab:
            decoded_tokens.append(inv_vocab[token_id])
        else:
            print(f"Token ID {token_id} not found in `inv_vocab`.")

    # Print the decoded tokens
    print(f"Decoded Tokens: {decoded_tokens}")

    # Return the decoded string
    return " ".join(decoded_tokens)


for idx, ids in enumerate(df_train['input_ids'][:5]):
    print(f"Row {idx + 1} Input IDs: {ids}")
    decoded = decode_ids(ids, inv_vocab)
    print(f"Row {idx + 1} Decoded Output: {decoded}")
# Looks about right.

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Replace -100 with <PAD> in labels
    if isinstance(labels_ids, np.ndarray):
        labels_ids[labels_ids == -100] = vocab['<PAD>']  # Handle NumPy arrays directly
    elif isinstance(labels_ids, list):
        labels_ids = [np.where(np.array(seq) == -100, vocab['<PAD>'], seq).tolist() for seq in labels_ids]
    else:
        raise ValueError("labels_ids must be a list or NumPy array.")

    # Decode predictions and labels (convert arrays to lists if necessary)
    pred_str = [decode_ids(ids.tolist() if isinstance(ids, np.ndarray) else ids, inv_vocab) for ids in pred_ids]
    label_str = [decode_ids(ids.tolist() if isinstance(ids, np.ndarray) else ids, inv_vocab) for ids in labels_ids]

    # Calculate Token Error Rate (TER)
    try:
        metric = WordErrorRate()
        ter = metric(pred_str, label_str).item()
    except Exception as e:
        return {"error": f"Error during TER calculation: {str(e)}"}

    # Log TER to wandb
    # wandb.log({"Token Error Rate (TER)": ter})

    return {"ter": ter}



print(compute_metrics) 

#region Testing evaluation metric
test_ids = [0, 2, 3, 4] 
decoded_str = decode_ids(test_ids, inv_vocab)
print("Decoded string:", decoded_str) # Looks oka
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


class MockPrediction:
    def __init__(self, predictions, label_ids):
        """
        Initialize with predictions and label IDs.
        :param predictions: List of predicted token ID sequences.
        :param label_ids: List of ground truth token ID sequences.
        """
        self.predictions = predictions
        self.label_ids = label_ids

mock_pred = MockPrediction(
    predictions=[[1, 4, 5, 2], [1, 6, 2, 0]],
    label_ids=[[1, 4, 5, 2], [1, 6, 5, -100]]
)

verify_alignment(mock_pred, inv_vocab, num_samples=2)
print("Decoded from [0]:", decode_ids([0], inv_vocab))

metrics = compute_metrics(mock_pred)
print("Metrics:", metrics)
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
wandb.init(project="master_thesis_finetuning", name="nineth_try")

# Update the model/num_parameters key with allow_val_change=True
wandb.config.update({"model/num_parameters": model.num_parameters()}, allow_val_change=True) # Relikt - als es noch der first_try war, der oft gefailt ist

# Training arguments
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    eval_strategy="steps",
    eval_steps=500, 
    num_train_epochs=epochs,  
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    save_steps=1000,  
    output_dir=output_dir,  
    logging_dir='./logs',
    logging_steps=logging_steps,  
    report_to="wandb",
    save_total_limit=2,  # Keep last 2 saved models
    load_best_model_at_end=True,  
    metric_for_best_model="ter",  
    greater_is_better=False, 
)

# Load the datasets and define the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_with_images,
    eval_dataset=val_dataset_with_images, 
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    callbacks=[ProgressPrintCallback()] 
)
#endregion

#region Training & Evaluation
# Start training with wandb logging enabled
trainer.train()

# Run final evaluation
final_results = trainer.evaluate(test_dataset_with_images)
print("Final Evaluation Results:", final_results)

# Log final TER to wandb
final_ter = final_results.get("eval_ter", "N/A")
wandb.log({"Final Token Error Rate (TER)": final_ter})



# Finish the wandb run
wandb.finish()
#endregion
#endregion







