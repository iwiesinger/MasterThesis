######### Preprocessing #########
# Here: 85/15 train/test split (-> no validation dataset)
# Also removed the _nx ending of variables


#region Read and get overview over raw data
#JSON file path
import json
file_path = 'MasterThesis/data/Akkadian.json'

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
df_raw_nx.reset_index(drop=True, inplace=True)
#endregion

#region Implement train- and test split: 0.7 training data, 0.15 validation data, 0.15 test data
random_seed = 42
df_shuffled = df_tok.sample(frac=1, random_state = random_seed).reset_index(drop=True)
print(df_shuffled.head())

def train_val_test_split(df):
    train_split = int(0.85*len(df))
    df_train = df[:train_split]
    df_test= df[train_split:]
    return df_train, df_test

df_train, df_test = train_val_test_split(df_shuffled)
print(df_train.head())
#endregion


######### Preparation for further analysis #########


#region Create Vocabulary and inversed vocabulary
from collections import Counter

# Flatten the list of tokenized signs
all_tokens = [token for sublist in df_tok['tok_signs'] for token in sublist]
print(all_tokens[9])

# Count the frequency of each token
token_counts = Counter(all_tokens)

# Create a vocabulary with token to index mapping
# Reserve indices 0-1 for special tokens
vocab = {token: idx for idx, (token, _) in enumerate(token_counts.items(), start=2)}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1


# Invert the vocabulary dictionary for decoding (if needed)
inv_vocab = {idx: token for token, idx in vocab.items()}
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


# Convert input_ids to labels, setting [PAD] tokens to -100
df_train['labels'] = df_train['input_ids'].apply(lambda ids: tokens_to_labels(ids, pad_token_id=vocab['<PAD>']))
df_test['labels'] = df_test['input_ids'].apply(lambda ids: tokens_to_labels(ids, pad_token_id=vocab['<PAD>']))

print(df_train[['input_ids', 'attention_mask', 'labels']].head())
print(df_train['attention_mask'][15])
print(df_train['input_ids'][15])
print(df_train['labels'][15])
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

# Create the dataloaders without X
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=20)
#endregion

#region Perplexity Callback
from transformers import TrainerCallback, TrainingArguments, Trainer, BertConfig, BertForMaskedLM
import math

class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        logs = kwargs.get("metrics", {})
        eval_loss = logs.get("eval_loss")
        if eval_loss is not None:
            perplexity = math.exp(eval_loss)
            print(f"Perplexity: {perplexity}")
#endregion


######### LM Training: BertLMHead #########

#region BertLMHead continuing model training
#### BERTLMHEAD

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
print("Test Perplexity: ", test_perplexity_nx)
#endregion

#region NEW BertLMHead
import wandb
from transformers import BertLMHeadModel, Trainer, TrainingArguments, BertConfig
from transformers.integrations import WandbCallback
import torch
import os

torch.cuda.memory_summary(device=None, abbreviated=False)
torch.cuda.empty_cache()
torch.cuda.is_available()

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize Weights and Biases for a new training run in the master_thesis project
wandb.init(project="master_thesis", name="pretraining_train_test_run")

# Load the base pre-trained BERT model and its configuration
config = BertConfig.from_pretrained('bert-base-uncased')
config.is_decoder = True
model = BertLMHeadModel.from_pretrained('bert-base-uncased', config=config)

# Create a folder to save models during this run
output_dir = 'MasterThesis/model_results_pretraining_train_test'

# Define the training arguments
training_args = TrainingArguments(
    output_dir=output_dir,             
    num_train_epochs=12,               
    per_device_train_batch_size=20,    
    per_device_eval_batch_size=20,
    warmup_steps=400,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy="no",          
    save_strategy="epoch",            
    report_to="wandb",                 
    load_best_model_at_end=False,
    fp16=True      
)

# Initialize the Trainer using train and test datasets (without validation dataset)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,       
    eval_dataset=test_dataset,         
    callbacks=[WandbCallback()]        
)

torch.cuda.empty_cache()

# Train the model
trainer.train()

# Evaluate the model on the test dataset
test_result = trainer.evaluate(eval_dataset=test_dataset)
print("Test Loss: ", test_result['eval_loss'])

import math
test_perplexity = math.exp(test_result['eval_loss'])
print("Test Perplexity: ", test_perplexity)

# End wandb logging
wandb.finish()
#endregion

#region Testing existing BertLMHead on test data with padding tokens == -100
from transformers import BertLMHeadModel, Trainer, TrainingArguments
import math

# Load the trained model from the checkpoint
model_path = "/home/ubuntu/MasterThesis/model_results_pretraining_train_test/checkpoint-11196/"
model = BertLMHeadModel.from_pretrained(model_path)

# Re-create the test dataset with the updated TransliterationDataset class
test_dataset = TransliterationDataset(df_test) 

# Define evaluation arguments
training_args = TrainingArguments(
    output_dir=model_path,
    per_device_eval_batch_size=20,
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

#endregion

#endregion
