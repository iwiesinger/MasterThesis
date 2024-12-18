######### Preprocessing #########
# Here: 85/15 train/test split (-> no validation dataset)
# Also removed the _nx ending of variables


#region Read and get overview over raw data
#JSON file path
import json
file_path = '/home/ubuntu/MasterThesis/data/Akkadian.json'

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
#endregion

#region Tokenizing the data and removing duplicates

# Filter out X and Newline
def tokenize_signs_exc_x(signs):
    signs = signs.replace('\n', ' <NEWLINE> ')  # Replace newline with special token
    tokens = signs.split()  # Split signs by whitespace
    tokens = ['<BOS>'] + [token for token in tokens if token not in ['X', '<NEWLINE>']] + ['<EOS>']  # Filter out 'X' and '<NEWLINE>'
    return tokens


df_tok = df_raw.copy()
df_tok['tok_signs'] = df_tok['signs'].apply(tokenize_signs_exc_x)
df_tok = df_tok.drop_duplicates(subset=['tok_signs'])
print(df_tok.head())
print(len(df_raw)) # 22054
print(len(df_tok)) # 21953 -> 61 rows removed.
print(type(df_raw))
print(df_tok.columns)
print(type(df_tok['tok_signs']))
print(df_tok['tok_signs'])


# Calculate the length of each entry in 'tok_signs'
df_tok['tok_signs_length'] = df_tok['tok_signs'].apply(len)

# Get summary statistics
summary_stats = df_tok['tok_signs_length'].describe(percentiles=[0.25, 0.5, 0.75, 0.8, 0.9])
print(summary_stats)
#mean       106.462737
#std        203.128378
#min          3.000000
#25%         19.000000
#50%         44.000000
#75%        108.000000
#max       3899.000000
#endregion

#region Removing uninformative rows
# sets of uninformative tokens
uninformative_tokens = {'<BOS>', '<NEWLINE>', '<EOS>', 'X'}

# Function to check if a row contains only uninformative tokens
def is_informative(tokens):
    return not all(token in uninformative_tokens for token in tokens)

# Filter rows to include only informative tokens
df_tok = df_tok[df_tok['tok_signs'].apply(is_informative)]
print(len(df_tok)) #21953 -> nothing changed.

# Find rows where 'tok_signs' is empty
empty_tok_signs = df_tok[df_tok['tok_signs'].apply(lambda tokens: len(tokens) == 0 if isinstance(tokens, list) else True)]
print(empty_tok_signs)
# none left :-) so all is tokenized nicely!

# Reset index 
df_tok.reset_index(drop=True, inplace=True)
#endregion

#region Removing rows that contain unsure tokens
# check if a token contains "/"
# Function to check if a token contains "/"
def contains_slash(token):
    return '/' in token

# Function to process a list of tokens and determine 'contains_unsure' and 'num_unsure'
def process_unsure_tokens(token_list):
    unsure_tokens = [token for token in token_list if contains_slash(token)]
    contains_unsure = 1 if unsure_tokens else 0
    num_unsure = len(unsure_tokens)
    return contains_unsure, num_unsure

df_tok['contains_unsure'], df_tok['num_unsure'] = zip(*df_tok['tok_signs'].apply(process_unsure_tokens))
df_tok['unknown_percent'] = df_tok['num_unsure'] / df_tok['tok_signs_length']
df_wo_unknowns = df_tok[df_tok['unknown_percent']<0.2]
print(df_wo_unknowns['unknown_percent'].describe())
print(len(df_wo_unknowns))
# 21527 rows are left! YEY!

# Removing unsure tokens
# Function to remove tokens containing "/"
def remove_unsure_tokens(token_list):
    return [token for token in token_list if '/' not in token]

df_wo_unknowns['tok_signs'] = df_wo_unknowns['tok_signs'].apply(remove_unsure_tokens)


#endregion

#region NOT YET ADJUSTED Implement train- and test split: 0.85 training data, 0.15 test data
'''random_seed = 42
df_shuffled = df_tok.sample(frac=1, random_state = random_seed).reset_index(drop=True)
print(df_shuffled.head())

def train_val_test_split(df):
    train_split = int(0.85*len(df))
    df_train = df[:train_split]
    df_test= df[train_split:]
    return df_train, df_test

df_train, df_test = train_val_test_split(df_shuffled)
print(df_train.head())'''
#endregion

#region Implement train- and test split: 0.70 training, 0.15 validation and 0.15 test data
random_seed = 42
df_shuffled = df_wo_unknowns.sample(frac=1, random_state=random_seed).reset_index(drop=True)
print(df_shuffled.head())

# Updated train-validation-test split function
def train_val_test_split(df):
    # Define the split indices
    train_split = int(0.70 * len(df))
    val_split = int(0.85 * len(df))  # 75% + 15% = 90%

    # Perform the splits
    df_train = df[:train_split]
    df_val = df[train_split:val_split]
    df_test = df[val_split:]

    return df_train, df_val, df_test

# Apply the function
df_train, df_val, df_test = train_val_test_split(df_shuffled)

# Print the sizes for verification
print(f"Train set size: {len(df_train)}")
print(f"Validation set size: {len(df_val)}")
print(f"Test set size: {len(df_test)}")
#endregion

######### Preparation for further analysis #########


#region Create Vocabulary and inversed vocabulary
from collections import Counter

# Flatten the list of tokenized signs
all_tokens = [token for sublist in df_shuffled['tok_signs'] for token in sublist]
print(all_tokens[9])

# Count the frequency of each token
token_counts = Counter(all_tokens)

# Create a vocabulary with token to index mapping
# Reserve indices 0-1 for special tokens
vocab = {token: idx for idx, (token, _) in enumerate(token_counts.items(), start=2)}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1
print(len(vocab))


# Invert the vocabulary dictionary for decoding (if needed)
inv_vocab = {idx: token for token, idx in vocab.items()}


# Save vocab to a JSON file
'''vocab_path = '/home/ubuntu/MasterThesis/code/excluding_unsure_tokens/vocab.json'
inv_vocab_path = '/home/ubuntu/MasterThesis/code/excluding_unsure_tokens/inv_vocab.json'

with open(vocab_path, 'w', encoding='utf-8') as f:
    json.dump(vocab, f, ensure_ascii=False, indent=4)

with open(inv_vocab_path, 'w', encoding='utf-8') as f:
    json.dump(inv_vocab, f, ensure_ascii=False, indent=4)

print(f"Vocabulary and inverse vocabulary saved at:\n{vocab_path}\n{inv_vocab_path}")'''

#endregion

#region Convert tokenized data to input IDs and attention masks
def tokens_to_ids(tokens, vocab, max_len=512):
    token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    token_ids = token_ids[:max_len]  # Truncate to max_len
    attention_mask = [1] * len(token_ids)
    padding_length = max_len - len(token_ids)
    token_ids = token_ids + [vocab['<PAD>']] * padding_length
    attention_mask = attention_mask + [0] * padding_length
    return token_ids, attention_mask

def tokens_to_labels(token_ids, pad_token_id=0):
    # [PAD] tokens -> -100 in the labels
    return [id if id != pad_token_id else -100 for id in token_ids]


# Create input IDs and attention masks for both datasets
df_train['input_ids'], df_train['attention_mask'] = zip(*df_train['tok_signs'].apply(lambda x: tokens_to_ids(x, vocab)))
df_test['input_ids'], df_test['attention_mask'] = zip(*df_test['tok_signs'].apply(lambda x: tokens_to_ids(x, vocab)))
df_val['input_ids'], df_val['attention_mask'] = zip(*df_val['tok_signs'].apply(lambda x: tokens_to_ids(x, vocab)))


# Convert input_ids to labels, setting [PAD] tokens to -100
df_train['labels'] = df_train['input_ids'].apply(lambda ids: tokens_to_labels(ids, pad_token_id=vocab['<PAD>']))
df_test['labels'] = df_test['input_ids'].apply(lambda ids: tokens_to_labels(ids, pad_token_id=vocab['<PAD>']))
df_val['labels'] = df_val['input_ids'].apply(lambda ids: tokens_to_labels(ids, pad_token_id=vocab['<PAD>']))

#endregion

#region Saving training, validation and test dataset for finetuning
# Function to save DataFrame to JSON
'''def save_to_json(df, folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    print(file_path)
    # Convert the DataFrame to a dictionary and save as JSON
    data = df.to_dict(orient='records')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Subfolder path
data_folder = '/home/ubuntu/MasterThesis/code/excluding_unsure_tokens/'

# Save each dataset to the subfolder
save_to_json(df_train, data_folder, 'df_train.json')
save_to_json(df_val, data_folder, 'df_val.json')
save_to_json(df_test, data_folder, 'df_test.json')

print(f"Datasets saved in the '{data_folder}' subfolder.")'''
#endregion

#region Create PyTorch Datasets and DataLoaders
import torch
from torch.utils.data import Dataset, DataLoader

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
train_dataset = TransliterationDataset(df_train)
test_dataset = TransliterationDataset(df_test)
val_dataset = TransliterationDataset(df_val)

# Create the dataloaders without X
train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=30)
val_loader = DataLoader(val_dataset, batch_size = 30, shuffle = True)
#endregion

#region Perplexity Callback
'''from transformers import TrainerCallback, TrainingArguments, Trainer, BertConfig, BertForMaskedLM
import math

class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        logs = kwargs.get("metrics", {})
        eval_loss = logs.get("eval_loss")
        if eval_loss is not None:
            perplexity = math.exp(eval_loss)
            print(f"Perplexity: {perplexity}")'''
#endregion

#region Perplexity Callback for wandb Logging
from transformers import TrainerCallback
import math

class PerplexityLoggingCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        # Extract the metrics logged during evaluation
        metrics = kwargs.get("metrics", {})
        eval_loss = metrics.get("eval_loss")

        if eval_loss is not None:
            perplexity = math.exp(eval_loss)
            # Log perplexity to wandb
            wandb.log({"epoch": state.epoch, "perplexity": perplexity})
            print(f"Epoch {state.epoch}: Perplexity = {perplexity}")
#endregion

#region Early Stopping Callback
from transformers import EarlyStoppingCallback

# Add EarlyStoppingCallback with a tolerance of 5 epochs
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=5  # Number of epochs to wait for improvement
)
#endregion

######### LM Training: BertLMHead #########

#region BertLMHead continuing model training
#### BERTLMHEAD
'''
import wandb
from transformers import BertLMHeadModel, Trainer, TrainingArguments, BertTokenizer, BertConfig
from transformers.integrations import WandbCallback

torch.cuda.memory_summary(device=None, abbreviated=False)
torch.cuda.empty_cache()

# Resume old WANDB Training
import wandb

api = wandb.Api()
runs = api.runs(path="iwiesinger-ludwig-maximilianuniversity-of-munich2357/master_thesis")
for i in runs:
  print("run name = ",i.name," id: ", i.id)


# Initialize Weights and Biases
wandb.init(project="master_thesis", id="k330z7hw", resume="must")

# Load the pre-trained BERT model and tokenizer
config = BertConfig.from_pretrained('bert-base-uncased')
config.is_decoder = True
model = BertLMHeadModel.from_pretrained('bert-base-uncased', config=config)

best_model_path = './results_2noNEWLINE/checkpoint-2564'  
model = BertLMHeadModel.from_pretrained(best_model_path)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results_2noNEWLINE',
    num_train_epochs=25,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=24,
    warmup_steps=300,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to="wandb",
    load_best_model_at_end=True  
)

# Initialize the Trainer
trainer_nx = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_nx,
    eval_dataset=val_dataset_nx,
    callbacks=[WandbCallback(), PerplexityCallback()]  
)

# Update the Trainer with the best model
trainer_nx.model = model

# Train the model
trainer_nx.train()
trainer_nx.load_best_model_at_end = True


# Evaluate the model on the test dataset
test_result_nx = trainer_nx.evaluate(eval_dataset=test_dataset_nx)
print("Test Loss: ", test_result_nx['eval_loss'])

import math
test_perplexity_nx = math.exp(test_result_nx['eval_loss'])
print("Test Perplexity: ", test_perplexity_nx)'''
#endregion

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

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize Weights and Biases for a new training run in the master_thesis project
wandb.init(project="master_thesis", name="pretrain_even_less_wd_higher_batch")

# Load the base pre-trained BERT model and its configuration
config = BertConfig.from_pretrained('bert-base-uncased')
config.is_decoder = True
config.vocab_size = len(vocab)
model = BertLMHeadModel.from_pretrained('bert-base-uncased', config=config, ignore_mismatched_sizes=True)
model.resize_token_embeddings(len(vocab))

# Create a folder to save models during this run
output_dir = '/home/ubuntu/MasterThesis/code/excluding_unsure_tokens/even_less_wd_with_higherbatch/'

# Define the training arguments
training_args = TrainingArguments(
    output_dir=output_dir,             # Output directory for saving models
    num_train_epochs=30,               # Train for 20 epochs
    per_device_train_batch_size=30,    # Increased batch size
    per_device_eval_batch_size=30,     # Match evaluation batch size
    warmup_steps=400,                  # Warmup steps for the learning rate scheduler
    weight_decay=0.0001,                 # Weight decay for regularization
    logging_dir='./logs',              # Directory for logs
    logging_steps=500,                 # Log every 500 steps to monitor training
    eval_steps=500,                    # Evaluate every 500 steps
    eval_strategy="steps",            # Perform evaluation during training at set intervals
    save_strategy="steps",            # Save models during training at set intervals
    save_steps=500,                    # Save every 500 steps
    save_total_limit=2,                # Keep the most recent and the best model
    metric_for_best_model="eval_loss",# Monitor validation loss to decide the best model
    greater_is_better=False,           # Lower validation loss is better
    report_to="wandb",                # Log training progress to wandb
    load_best_model_at_end=True,       # Load the best model at the end of training
    fp16=True                          # Use mixed precision for faster training
)

# Initialize the Trainer using train and test datasets (without validation dataset)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,       # Train on the train dataset
    eval_dataset=val_dataset,   # Evaluate on the validation dataset
    callbacks=[
        WandbCallback(), 
        PerplexityLoggingCallback(), 
        early_stopping_callback]
) 


# Train the model
trainer.train()

# Evaluate the model on the test dataset
test_result = trainer.evaluate(eval_dataset=test_dataset)
print("Test Loss: ", test_result['eval_loss'])

import math
test_perplexity = math.exp(test_result['eval_loss'])
print("Test Perplexity: ", test_perplexity)

# Save the trained model to output_dir
trainer.save_model(output_dir)
print(f"Model saved to {output_dir}")

# End wandb logging
wandb.finish()
#endregion

#region Testing existing BertLMHead on test data with padding tokens == -100
'''from transformers import BertLMHeadModel, Trainer, TrainingArguments
import math

# Load the trained model from the checkpoint
model_path = "/home/ubuntu/MasterThesis/results_pretrain_20241127_noval/checkpoint-8333/"
model = BertLMHeadModel.from_pretrained(model_path)

# Re-create the test dataset with the updated TransliterationDataset class
test_dataset = TransliterationDataset(df_test) 

# Define evaluation arguments
training_args = TrainingArguments(
    output_dir=model_path,
    per_device_eval_batch_size=24,
    logging_dir='./logs',
    report_to="none"  # Disable logging to wandb or other services during evaluation
)

# Initialize the Trainer with only evaluation dataset
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset
)

# Run evaluation on the test dataset
test_result = trainer.evaluate()
print("Test Loss: ", test_result['eval_loss'])

# Calculate perplexity from the evaluation loss
test_perplexity = math.exp(test_result['eval_loss'])
print("Test Perplexity: ", test_perplexity)
'''
#endregion
