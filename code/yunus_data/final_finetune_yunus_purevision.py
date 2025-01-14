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
#pretrained_bert_path = '/home/ubuntu/MasterThesis/code/yunus_data/pretraining_after_better_reorder/checkpoint-840/'
output_dir = '/home/ubuntu/MasterThesis/code/yunus_data/finetuning_PUREVISION'
#safetensors_file = pretrained_bert_path + "model.safetensors"
train_data_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_art.json'
finetune_data_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_train_val_aug.json'
test_data_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_test_resized.json'
#val_data_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_val_big_aug.json'
#vocab_path = '/home/ubuntu/MasterThesis/code/yunus_data/vocab.json'
#inv_vocab_path = '/home/ubuntu/MasterThesis/code/yunus_data/inv_vocab.json'
#endregion


#region Import datasets and (inv) vocab from pretraining
with open(train_data_path, 'r') as f:
    df_pretrain = pd.DataFrame(json.load(f))

with open(finetune_data_path, 'r') as f:
    df_finetune = pd.DataFrame(json.load(f))

with open(test_data_path, 'r') as f:
    df_test = pd.DataFrame(json.load(f))

#with open(vocab_path, 'r') as f:
#    vocab = json.load(f)

#with open(inv_vocab_path, 'r') as f:
#    inv_vocab = json.load(f)
#inv_vocab = {int(k): v for k, v in inv_vocab.items()}

#for key, value in list(vocab.items())[:5]:
#    print(f"{key}: {value}")
#vocab = {key: int(value) for key, value in vocab.items()}
#endregion


#region NEW Custom Class + Dataset Creation for treatment without vocab
# use tokenizer instead of vocab
class TransliterationWithImageDataset(Dataset):
    def __init__(self, root_dir, df, feature_extractor, tokenizer, max_seq_len=512):
        self.root_dir = root_dir
        self.df = df
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer # instead of vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        while True:
            img_name = self.df.iloc[idx]['img_name']
            image_path = os.path.join(self.root_dir, img_name)

            # Check if image exists
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}. Skipping row {idx}.")
                idx = (idx + 1) % len(self.df)
                continue

            try:
                image = Image.open(image_path).convert("RGB")
                pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values.squeeze()
            except Exception as e:
                print(f"Error processing image {image_path}: {e}. Skipping row {idx}.")
                idx = (idx + 1) % len(self.df)
                continue

            # input ids and attention masks
            input_ids = torch.tensor(self.df.iloc[idx]['input_ids'])
            attention_mask = torch.tensor(self.df.iloc[idx]['attention_mask'])

            # use tokenizer again
            labels = input_ids.clone()
            labels[input_ids == self.tokenizer.pad_token_id] = -100

            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }



from transformers import AutoTokenizer

#tokenizer and feature extractor
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224")

# custom datasets and data loaders
pretrain_dataset_with_images = TransliterationWithImageDataset(root_dir=root_dir_pretrain, df=df_pretrain, feature_extractor=feature_extractor, tokenizer=tokenizer)
finetune_dataset_with_images = TransliterationWithImageDataset(root_dir=root_dir_finetune, df=df_finetune, feature_extractor=feature_extractor, tokenizer=tokenizer)
test_dataset_with_images = TransliterationWithImageDataset(root_dir=root_dir_test, df=df_test, feature_extractor=feature_extractor, tokenizer=tokenizer)

pretrain_loader_with_images = DataLoader(pretrain_dataset_with_images, batch_size=10, shuffle=True)
finetune_loader_with_images = DataLoader(finetune_dataset_with_images, batch_size=10, shuffle=True)
test_loader_with_images = DataLoader(test_dataset_with_images, batch_size=10)

#endregion

# Print dataset sizes
print('Number of training examples:', len(pretrain_dataset_with_images))  # Pretraining
print('Number of fine-tuning examples:', len(finetune_dataset_with_images))  # Fine-tuning 
print('Number of test examples:', len(test_dataset_with_images))  # Test

#region OLD progress print callback for validation dataset and early stopping callback
# I don't dare throwing this away
'''
#region ProgressPrintCallback and Early Stopping Callback
class ProgressPrintCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current_step = state.global_step
        current_epoch = state.epoch
        ter = metrics.get("eval_ter", "N/A")  # Get TER from metrics

        # Log TER to wandb if it's a valid metric
        if ter != "N/A":
            wandb.log({"Token Error Rate (TER)": ter, "Step": current_step, "Epoch": current_epoch})

        print(f"Step {current_step}, Epoch {current_epoch}, TER: {ter}")


# Add EarlyStoppingCallback with patience
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=15,  
    early_stopping_threshold=0.001  
)
'''
#endregion



#region #### Model Setup ####

bert_config = BertConfig.from_pretrained('bert-base-uncased')
bert_config.is_decoder = True 
bert_config.add_cross_attention = True  # Enable cross-attention for decoder
#bert_config.vocab_size = 124 
#print("Standard vocab size:", bert_config.vocab_size)
swin_config = SwinConfig()

# SwinBERT 
config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(swin_config, bert_config)
model = VisionEncoderDecoderModel(config=config)
model.encoder = SwinModel(swin_config)
model.decoder = BertLMHeadModel.from_pretrained('bert-base-uncased', config=bert_config)
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
# Special Token IDs - according to TOKENIZER
model.config.pad_token_id = tokenizer.pad_token_id
model.config.decoder_start_token_id = tokenizer.cls_token_id  
model.config.eos_token_id = tokenizer.sep_token_id  
model.config.unk_token_id = tokenizer.unk_token_id


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
epochs = 20*1
batch_size = 10
#eval_steps = np.round(len(df_train) / batch_size * epochs / 20, 0)
logging_steps = np.round(len(df_pretrain) / batch_size * epochs / 20, 0)  
#endregion
# Initialize missing layers (cross-attention)
model.decoder.apply(model.decoder._init_weights)  # Randomly initializes only the missing layers


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

# updated with tokenizer
def decode_ids(ids, tokenizer):
    return tokenizer.decode(ids, skip_special_tokens=True)


def compute_metrics(pred):
    # Debugging: Print pred keys and shapes
    print(f"Pred object keys: {dir(pred)}")
    print(f"Predictions type: {type(pred.predictions)}")
    print(f"Label IDs type: {type(pred.label_ids)}")

    # Predictions and labels shape
    print(f"Predictions shape: {pred.predictions.shape}")
    print(f"Label IDs shape: {pred.label_ids.shape}")

    # First few raw predictions and labels
    print(f"Predictions (raw): {pred.predictions[:1]}")
    print(f"Labels (raw): {pred.label_ids[:1]}")

    # Replace -100 with pad_token_id in labels
    labels_ids = pred.label_ids
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = [tokenizer.decode(ids, skip_special_tokens=True) for ids in pred.predictions]
    label_str = [tokenizer.decode(ids, skip_special_tokens=True) for ids in labels_ids]

    # Debugging: Print decoded predictions and labels
    print(f"Decoded Predictions: {pred_str[:5]}")
    print(f"Decoded Labels: {label_str[:5]}")
    print(f"Length of predictions: {len(pred_str)}")
    print(f"Length of labels: {len(label_str)}")

    try:
        # Initialize WordErrorRate
        wer_metric = WordErrorRate()

        # Update with predictions and labels
        wer_metric.update(pred_str, label_str)

        # Compute TER
        ter = wer_metric.compute()
        return {"eval_ter": ter.item()}
    except Exception as e:
        # Return error for debugging
        print(f"Error during TER calculation: {str(e)}")
        return {"eval_error": f"Error during TER calculation: {str(e)}"}


#endregion

'''#region verify alignment
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
#endregion'''


#region #### Training ####
#region Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.empty_cache() # memory <3
#torch.cuda.reset_peak_memory_stats()
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
model.to(device)

# Initialize wandb
wandb.init(project="large_data_pretraining", name="vision_only_20epochs20batch_try1")
wandb.config.update({"model/num_parameters": model.num_parameters()}, allow_val_change=True) 


#region Pretraining
pretraining_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,               # generation during evaluation
    eval_strategy="no",                       # no evaluation during pretraining
    num_train_epochs=20,                  
    per_device_train_batch_size=batch_size,  
    fp16=True,                                # mixed precision 
    save_strategy="epoch",                    # save at end of epoch
    save_total_limit=1,                       # keep one checkpoint (last one)
    output_dir=output_dir + "/pretraining",  
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
pretrained_model_path = output_dir + "/pretrained_model"
pretrained_model_path = '/home/ubuntu/MasterThesis/code/yunus_data/finetuning_PUREVISION/pretraining/checkpoint-65100'
model.save_pretrained(pretrained_model_path)
wandb.finish()
#endregion

#region Finetuning

wandb.init(project="large_data_finetuning", name="finetune_no_validation_try2_workingTER")
wandb.config.update({"model/num_parameters": model.num_parameters()}, allow_val_change=True)

#region Finetuning
# Load the pretrained model
model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_path)

# training arguments for finetuning
finetuning_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,               # generation
    eval_strategy="no",                       # no eval
    num_train_epochs=15,                      
    per_device_train_batch_size=batch_size,  
    fp16=True,                               
    save_strategy="epoch",                    
    save_total_limit=1,                      # 1 checkpoint
    output_dir=output_dir + "/fine_tuning",  
    logging_dir='./logs_finetuning',         
    logging_steps=logging_steps,             
    report_to="wandb",                       
)

# trainer for finetuning
finetuning_trainer = Seq2SeqTrainer(
    model=model,                              # pretrained model
    args=finetuning_args,                     # Fine-tuning arguments
    train_dataset=finetune_dataset_with_images,  # Second dataset for fine-tuning
    data_collator=default_data_collator, 
    compute_metrics=compute_metrics,          
)

# Fine-tune the model on the second dataset
finetuning_trainer.train()

fine_tuned_model_path = output_dir + "/fine_tuned_model"
model.save_pretrained(fine_tuned_model_path)

test_results = finetuning_trainer.evaluate(test_dataset_with_images)

final_ter = test_results.get("eval_ter", None)
if final_ter is not None:
    wandb.log({"Test Token Error Rate (TER)": final_ter})
else:
    print("TER could not be calculated. Check the evaluation results:", test_results)


wandb.finish()
#endregion


#endregion







'''

#region Training arguments with validation dataset

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,              
    eval_strategy="epoch",                    
    num_train_epochs=epochs,                 
    per_device_train_batch_size=batch_size, 
    per_device_eval_batch_size=batch_size,    
    fp16=True,                               
    save_strategy="epoch",                    
    save_total_limit=2,                      
    load_best_model_at_end=True,              
    metric_for_best_model="eval_ter",        
    greater_is_better=False,                  
    output_dir=output_dir,          
    logging_dir='./logs',                   
    logging_steps=logging_steps,              
    report_to="wandb",                      
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_with_images,  # Training dataset
    eval_dataset=val_dataset_with_images,    # Validation dataset
    data_collator=default_data_collator,     # Data collator for batching
    compute_metrics=compute_metrics,         # Function to compute validation metrics (e.g., TER)
    callbacks=[ProgressPrintCallback(), early_stopping_callback],  # Add early stopping
)

#endregion
#endregion

#region Training Arguments without validation dataset

# Training arguments
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,       
    evaluation_strategy="no",         
    num_train_epochs=epochs,          
    per_device_train_batch_size=batch_size, 
    per_device_eval_batch_size=batch_size,  
    fp16=True,                       
    save_strategy="epoch",            
    output_dir=output_dir,             
    logging_dir='./logs',        
    logging_steps=logging_steps,      
    report_to="wandb",              
    save_total_limit=1,               
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_with_images,  
    eval_dataset=test_dataset_with_images,  
    data_collator=default_data_collator,    
    compute_metrics=compute_metrics,      
    callbacks=[ProgressPrintCallback()],    
)
#endregion

#region Training & Evaluation
trainer.train()

final_results = trainer.evaluate(test_dataset_with_images)


final_ter = final_results.get("eval_ter", None)  
if final_ter is not None:
    wandb.log({"Final Token Error Rate (TER)": final_ter})
else:
    print("TER could not be calculated. Check the evaluation results:", final_results)


wandb.finish()
#endregion
'''