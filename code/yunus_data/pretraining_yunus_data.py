######### Preprocessing #########
# Here: 85/15 train/test split (-> no validation dataset)
# Also removed the _nx ending of variables


#region Read and get overview over raw data
#JSON file path
import json
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

os.getcwd()

train_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_train_resized.json'
val_path = '/home/ubuntu/MasterThesis/code/yunus_data/df_val_resized.json'

with open(train_path, 'r') as f:
    train_raw = pd.DataFrame(json.load(f))

with open(val_path, 'r') as f:
    val_raw = pd.DataFrame(json.load(f))

# Length of the list
train_length = len(train_raw)
val_length = len(val_raw)
print(f"Train length: {train_length}, Validation length: {val_length}")
print(train_raw.columns)

#region OLD - not necessary to map image_name or classes to annotations anymore
# Add 'image_name' to annotations in coco_recog_train
train_image_id_to_name = {image['id']: image['file_name'] for image in coco_recog_train['images']}
for annotation in coco_recog_train['annotations']:
    annotation['image_name'] = train_image_id_to_name.get(annotation['image_id'], None)

# Add 'image_name' to annotations in coco_recog_val
val_image_id_to_name = {image['id']: image['file_name'] for image in coco_recog_val['images']}
for annotation in coco_recog_val['annotations']:
    annotation['image_name'] = val_image_id_to_name.get(annotation['image_id'], None)

classes_path = '/home/ubuntu/MasterThesis/code/yunus_data/classes.txt'

# Read and parse the classes.txt file
with open(classes_path, 'r') as file:
    class_list = eval(file.read()) 

# Function to update categories with 'abz'
def update_categories_with_abz(dataset, class_list):
    for category in dataset['categories']:
        category_id = category['id']
        if 0 <= category_id < len(class_list):
            category['abz'] = class_list[category_id]
        else:
            category['abz'] = None

# Update categories in coco_recog_train and coco_recog_val
update_categories_with_abz(coco_recog_train, class_list)
update_categories_with_abz(coco_recog_val, class_list)

#endregion
#endregion

#region OLD Split coco_recog_val into coco_recog_val (first 50 images) and coco_recog_test (last 50 images)

'''
random_seed = 42
df_shuffled = df_wo_unknowns.sample(frac=1, random_state=random_seed).reset_index(drop=True)
'''
'''
# Split images
val_images = coco_recog_val['images'][:50]
test_images = coco_recog_val['images'][50:]

# Get the corresponding annotations for validation and test
val_image_ids = {image['id'] for image in val_images}
test_image_ids = {image['id'] for image in test_images}
print(f"Validation image ids: {val_image_ids},\n Test image ids: {test_image_ids}")

val_annotations = [annotation for annotation in coco_recog_val['annotations'] if annotation['image_id'] in val_image_ids]
test_annotations = [annotation for annotation in coco_recog_val['annotations'] if annotation['image_id'] in test_image_ids]
print(type(val_annotations))
print(val_annotations[1])


# Keep the categories as is for both
categories = coco_recog_val['categories']

# Create the validation and test datasets
coco_recog_val_split = {
    'images': val_images,
    'annotations': val_annotations,
    'categories': categories
}

coco_recog_test = {
    'images': test_images,
    'annotations': test_annotations,
    'categories': categories
}

print(coco_recog_train['images'][1])
print(coco_recog_val['images'][1])'''

'''# Save the datasets to JSON files
val_output_path = '/home/ubuntu/coco_recog_val.json'
test_output_path = '/home/ubuntu/coco_recog_test.json'

with open(val_output_path, 'w') as val_file:
    json.dump(coco_recog_val_split, val_file, indent=4)

with open(test_output_path, 'w') as test_file:
    json.dump(coco_recog_test, test_file, indent=4)

print(f"Validation dataset saved to {val_output_path}")
print(f"Test dataset saved to {test_output_path}")
'''
#endregion


#region Create Pretraining dataframe

#region OLD Function to create a pretraining DataFrame
def create_pretraining_dataframe(coco_recog):
    # Create a mapping from category_id to ABZ notation
    category_mapping = {category["id"]: category.get("abz", None) for category in coco_recog["categories"]}

    # Initialize list to store rows for new dataframe
    rows = []

    # iterate hrough images in coco_recog
    for image in coco_recog["images"]:
        img_name = image["file_name"]
        height = image["height"]
        width = image["width"]

        # get annotations
        annotations = [ann for ann in coco_recog["annotations"] if ann["image_name"] == img_name]

        # details
        category_ids = [ann["category_id"] for ann in annotations]
        bboxes = [ann["bbox"] for ann in annotations]
        areas = [ann["area"] for ann in annotations]
        abz_notations = [category_mapping[cat_id] for cat_id in category_ids]

        # append data
        rows.append({
            "img_name": img_name,
            "height": height,
            "width": width,
            "category_id": category_ids,
            "bbox": bboxes,
            "area": areas,
            "abz": abz_notations
        })

    return pd.DataFrame(rows)
#endregion

pd.set_option("display.max_columns", None)

df_train = create_pretraining_dataframe(coco_recog_train)
print(df_train.head())

# For coco_recog_val_split
df_val = create_pretraining_dataframe(coco_recog_val)
print(df_val.head())

print(len(df_train))
print(len(df_val))
# For coco_recog_test
#df_test = create_pretraining_dataframe(coco_recog_test)
#print(df_test.head())
#endregion

pd.set_option('display.max_rows', None)       
pd.set_option('display.max_columns', None)    
pd.set_option('display.width', 1000)          
pd.set_option('display.max_colwidth', None) 

print(train_raw['img_name'].head())
print(train_raw.head(2))

#region OLD Create vocabulary
# Use categories as the vocabulary and inverse vocabulary
def create_vocab_and_inverse(df):
    vocab = {category["abz"]: category["id"] for category in df["categories"]}
    inv_vocab = {category["id"]: category["abz"] for category in df["categories"]}

    # special tokens
    max_id = max(vocab.values())
    vocab.update({"<BOS>": max_id + 1, "<EOS>": max_id + 2, "<UNK>": max_id + 3, "<PAD>": max_id + 4})
    inv_vocab.update({max_id + 1: "<BOS>", max_id + 2: "<EOS>", max_id + 3: "<UNK>", max_id + 4: "<PAD>"})

    return vocab, inv_vocab

vocab, inv_vocab = create_vocab_and_inverse(coco_recog_train)
print(vocab)
print()

# Saving
vocab_path = '/home/ubuntu/MasterThesis/code/yunus_data/vocab.json'
inv_vocab_path = '/home/ubuntu/MasterThesis/code/yunus_data/inv_vocab.json'

with open(vocab_path, 'w', encoding='utf-8') as f:
    json.dump(vocab, f, ensure_ascii=False, indent=4)

with open(inv_vocab_path, 'w', encoding='utf-8') as f:
    json.dump(inv_vocab, f, ensure_ascii=False, indent=4)

print(f"Vocabulary and inverse vocabulary saved at:\n{vocab_path}\n{inv_vocab_path}")


#endregion

#region NEW: Open Vocabulary and Inv Vocabulary
with open('/home/ubuntu/MasterThesis/code/yunus_data/vocab.json', "r") as vocab_file:
    vocab = json.load(vocab_file)
with open('/home/ubuntu/MasterThesis/code/yunus_data/inv_vocab.json', "r") as inv_vocab_file:
    inv_vocab = json.load(inv_vocab_file)
#endregion


# modify tokenization - including unknown tokens
def tokenize_with_special_tokens(df, column_name="abz"):
    df["tok_signs"] = df[column_name].apply(
        lambda x: ["<BOS>"] + [token if token in vocab else "<UNK>" for token in x] + ["<EOS>"]
    )
    return df

df_train = tokenize_with_special_tokens(train_raw, column_name="abz")
df_val = tokenize_with_special_tokens(val_raw, column_name="abz")

#df_test = tokenize_with_special_tokens(df_test, column_name="abz")
print(new_raw.columns)
print(df_train.columns)
df_new['img_name'] = df_new['_id'] + '.jpg'


# Inspect the updated DataFrame
print(df_train[["abz", "tok_signs"]].head(2))
#endregion

#endregion



######### Preparation for further analysis #########


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
df_val['input_ids'], df_val['attention_mask'] = zip(*df_val['tok_signs'].apply(lambda x: tokens_to_ids(x, vocab)))
df_val['input_ids'] = df_val['input_ids'].apply(list)  # Ensure input_ids is a list
df_val['attention_mask'] = df_val['attention_mask'].apply(list) 

print(type(df_train['input_ids']))

# Define the tokens_to_labels function
def tokens_to_labels(token_ids, pad_token_id=123):
    return [token_id if token_id != pad_token_id else -100 for token_id in token_ids]

# Apply tokens_to_labels to the 'input_ids' column for df_train
df_train["labels"] = df_train["input_ids"].apply(lambda ids: tokens_to_labels(ids, pad_token_id=vocab["<PAD>"]))
df_val["labels"] = df_val["input_ids"].apply(lambda ids: tokens_to_labels(ids, pad_token_id=vocab["<PAD>"]))

print(df_train.head())
print(len(df_train))
print(len(df_val))
print(df_train.head(2))
print(df_train.columns)
print(df_train[['input_ids', 'attention_mask', 'labels']].head(2))
print(df_train.columns)
print(len(df_train))
print(df_train_resized.columns)
#endregion

#region Saving training, validation and test dataset for finetuning
# Function to save DataFrame to JSON
def save_to_json(df, folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    print(file_path)
    # Convert the DataFrame to a dictionary and save as JSON
    data = df.to_dict(orient='records')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Subfolder path
data_folder = '/home/ubuntu/MasterThesis/code/yunus_data/'

save_to_json(df_train, data_folder, 'df_train_resized.json')
save_to_json(df_val, data_folder, 'df_val_resized.json')
#save_to_json(df_test, data_folder, 'df_test.json')

print(f"Datasets saved in the '{data_folder}' subfolder.")
#endregion

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
train_dataset = TransliterationDataset(df_train)
#test_dataset = TransliterationDataset(df_test)
val_dataset = TransliterationDataset(df_val)

# Create the dataloaders without X
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=30)
val_loader = DataLoader(val_dataset, batch_size = 10)
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
        # extract metrics during eval
        metrics = kwargs.get("metrics", {})
        eval_loss = metrics.get("eval_loss")

        if eval_loss is not None:
            perplexity = math.exp(eval_loss)
            # log
            wandb.log({"epoch": state.epoch, "perplexity": perplexity})
            print(f"Epoch {state.epoch}: Perplexity = {perplexity}")
#endregion

#region Early Stopping Callback
from transformers import EarlyStoppingCallback

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=5  # tolerance
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

#  setup gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

wandb.init(project="master_thesis", name="better_reordered_transliterations_try3_10batch")

config = BertConfig.from_pretrained('bert-base-uncased')
config.is_decoder = True
config.vocab_size = len(vocab)
model = BertLMHeadModel.from_pretrained('bert-base-uncased', config=config, ignore_mismatched_sizes=True)
model.resize_token_embeddings(len(vocab))

output_dir = '/home/ubuntu/MasterThesis/code/yunus_data/pretraining_after_better_reorder/'

# Define the training arguments
training_args = TrainingArguments(
    output_dir=output_dir,             
    num_train_epochs=15,             
    per_device_train_batch_size=10,    
    warmup_steps=400,                  
    weight_decay=0.001,                
    logging_dir='./logs',            
    logging_steps=500,                 
    save_strategy="epoch",             
    save_total_limit=1,               
    report_to="wandb",           
    fp16=True                         
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,      
    callbacks=[
        WandbCallback(), 
        PerplexityLoggingCallback()]
) 


trainer.train()

# evaluation
test_result = trainer.evaluate(eval_dataset=val_dataset)
print("Test Loss: ", test_result['eval_loss'])

import math
test_perplexity = math.exp(test_result['eval_loss'])
print("Test Perplexity: ", test_perplexity)

trainer.save_model(output_dir)
print(f"Model saved to {output_dir}")

# End wandb logging
wandb.finish()
#endregion

#region Testing existing BertLMHead on test data with padding tokens == -100
from transformers import BertLMHeadModel, Trainer, TrainingArguments
import math

# loading model
model_path = "/home/ubuntu/MasterThesis/code/excluding_unsure_tokens/results_higherbatch/checkpoint-6500/"
model = BertLMHeadModel.from_pretrained(model_path)

# Re-create the test dataset with the updated TransliterationDataset class
test_dataset = TransliterationDataset(df_val) 

# evaluation arguments
training_args = TrainingArguments(
    output_dir=model_path,
    per_device_eval_batch_size=24,
    logging_dir='./logs',
    report_to="none" 
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset
)

test_result = trainer.evaluate()
print("Test Loss: ", test_result['eval_loss'])

test_perplexity = math.exp(test_result['eval_loss'])
print("Test Perplexity: ", test_perplexity)

#endregion
