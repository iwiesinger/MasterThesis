######### Overview & Preprocessing #########

#region Read and get overview over raw data
#JSON file path
import json
file_path = 'Akkadian.json'

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
print(df_raw.head())
print(df_raw.columns) # id, script, signs
print(len(df_raw)) #22054 observations
print(type(df_raw))
print(type(df_raw['script']))
print(df_raw['script'].head())
#endregion

#region Variable Overviews
#region Period variable
# Looking at period
df_raw['period'] = df_raw['script'].apply(lambda x: x['period'])

# Get the unique values of the 'period' column
unique_periods = df_raw['period'].unique()
unique_periods_list = unique_periods.tolist()
print(unique_periods_list)
print(len(unique_periods_list))
#endregion

#region Period Counts
period_counts = df_raw['period'].value_counts()
print(period_counts)
#endregion

#region PeriodModifier variable
df_raw['periodModifier'] = df_raw['script'].apply(lambda x: x['periodModifier'])
unique_period_modifier = df_raw['periodModifier'].unique()
unique_period_modifier = unique_period_modifier.tolist()
print(unique_period_modifier)

#endregion

#region Uncertain
df_raw['uncertain'] = df_raw['script'].apply(lambda x: x['uncertain'])
uncertain = df_raw['uncertain'].unique()
uncertain = uncertain.tolist()
print(uncertain)
#endregion

#region sortKey
df_raw['sortKey'] = df_raw['script'].apply(lambda x: x['sortKey'])
sortKey = df_raw['sortKey'].unique()
sortKey = sortKey.tolist()
print(sortKey)
print(len(sortKey))
#endregion

#region _id
print(df_raw['_id'].head(5))
id_unique = df_raw['_id'].unique()
print(len(id_unique))
# ID is a unique identifier of each observation
id_list = list(id_unique)
#endregion
#endregion

#region Tokenizing the data and removing duplicates

# Filter out X and Newline
def tokenize_signs_exc_x(signs):
    signs = signs.replace('\n', ' <NEWLINE> ')  # Replace newline with special token
    tokens = signs.split()  # Split signs by whitespace
    tokens = ['<BOS>'] + [token for token in tokens if token not in ['X', '<NEWLINE>']] + ['<EOS>']  # Filter out 'X' and '<NEWLINE>'
    return tokens


df_raw_nx = df_raw.copy()
df_raw_nx['tok_signs'] = df_raw_nx['signs'].apply(tokenize_signs_exc_x)
df_raw_nx = df_raw_nx.drop_duplicates(subset=['tok_signs'])
print(df_raw_nx.head())
print(len(df_raw)) # 22054
print(len(df_raw_nx)) # 22004 -> 50 rows removed.
print(type(df_raw))
print(df_raw_nx.columns)
#endregion

#region Removing uninformative rows
# sets of uninformative tokens
uninformative_tokens_nx = {'<BOS>', '<NEWLINE>', '<EOS>'}

# Function to check if a row contains only uninformative tokens
def is_informative_nx(tokens):
    return not all(token in uninformative_tokens_nx for token in tokens)

# Filter rows
df_raw_nx = df_raw_nx[df_raw_nx['tok_signs'].apply(is_informative_nx)]
print(len(df_raw_nx)) #21994 -> 10 rows completely uninformative without X


# Reset the index if needed
df_raw_nx.reset_index(drop=True, inplace=True)
#endregion

#region How long are resulting rows?

#region Statistics
# Calculate basic statistics for the token counts
df_raw_nx['token_count'] = df_raw_nx['tok_signs'].apply(len)

# Get basic statistics like mean, median, percentiles, etc.
token_count_stats = df_raw_nx['token_count'].describe()

# Display the statistics
print(token_count_stats)
#endregion

#region Visualization

#region Big Picture
import pandas as pd
import matplotlib.pyplot as plt

# Create the token count column
df_raw_nx['token_count'] = df_raw_nx['tok_signs'].apply(len)

# Define the bins (ranges) for token counts, including the last bucket for >3500
bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, float('inf')]

# Define the labels for each bucket, with the last being ">3500"
labels = ['0-500', '500-1000', '1000-1500', '1500-2000', '2000-2500', '2500-3000', '3000-3500', '>3500']

# Create the bucketed token counts with labels
df_raw_nx['token_count_bucket'] = pd.cut(df_raw_nx['token_count'], bins=bins, labels=labels, right=False)

# Get the frequency distribution of the buckets
bucket_distribution = df_raw_nx['token_count_bucket'].value_counts().sort_index()

# Plot the bar chart
plt.figure(figsize=(10,8))
ax = bucket_distribution.plot(kind='bar')

# Add counts on top of each bar
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# Add labels and title
plt.title('Token Count Ranges')
plt.xlabel('Token Count Ranges')
plt.ylabel('Frequency (Number of Rows) within the range')
plt.xticks(rotation=45)
plt.show()
plt.savefig('plots/tok_count_ranges_total.jpg')

# Most Observations have less than 500 tokens. 
#endregion

#region Less than 500 Zoom-In

# Define the bins (ranges) for token counts between 0 and 500, in steps of 50
bins_0_500 = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

# Define the labels for each bucket
labels_0_500 = ['<50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350-400', '400-450', '450-500']

# Create the bucketed token counts with labels for the range 0-500
df_raw_nx['token_count_bucket_0_500'] = pd.cut(df_raw_nx['token_count'], bins=bins_0_500, labels=labels_0_500, right=False)

# Get the frequency distribution of the buckets for the range 0-500
bucket_distribution_0_500 = df_raw_nx['token_count_bucket_0_500'].value_counts().sort_index()

# Plot the bar chart
plt.figure(figsize=(10,8))
ax = bucket_distribution_0_500.plot(kind='bar')

# Add counts on top of each bar
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# Add labels and title
plt.title('Distribution of Token Count Buckets (0-500)')
plt.xlabel('Token Count Range')
plt.ylabel('Frequency (Number of Rows)')
plt.xticks(rotation=45)
plt.show()
plt.savefig('plots/less500_tok_count_ranges.jpg')
#endregion

#region Less than 50 Zoom-In
# Define the bins (ranges) for token counts less than 50, in steps of 10
bins_under_50 = [0, 10, 20, 30, 40, 50]

# Define the labels for each bucket under 50
labels_under_50 = ['<10', '10-20', '20-30', '30-40', '40-50']

# Create the bucketed token counts with labels for the range <50
df_raw_nx['token_count_bucket_under_50'] = pd.cut(df_raw_nx['token_count'], bins=bins_under_50, labels=labels_under_50, right=False)

# Get the frequency distribution of the buckets for the range <50
bucket_distribution_under_50 = df_raw_nx['token_count_bucket_under_50'].value_counts().sort_index()

# Plot the bar chart for the range <50
plt.figure(figsize=(10,8))
ax = bucket_distribution_under_50.plot(kind='bar')

# Add counts on top of each bar
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# Add labels and title
plt.title('Distribution of Token Count Buckets (<50)')
plt.xlabel('Token Count Range')
plt.ylabel('Frequency (Number of Rows)')
plt.xticks(rotation=45)
plt.show()
#endregion

#endregion

#endregion

#region Implement train- and test split: 0.7 training data, 0.15 validation data, 0.15 test data
random_seed = 42
df_shuffled_nx = df_raw_nx.sample(frac=1, random_state = random_seed).reset_index(drop=True)
print(df_shuffled_nx.head())

def train_val_test_split(df):
    train_split = int(0.7*len(df))
    val_split = int(0.85*len(df))
    df_train = df[:train_split]
    df_val = df[train_split:val_split]
    df_test= df[val_split:]
    return df_train, df_val, df_test

df_train_nx, df_val_nx, df_test_nx = train_val_test_split(df_shuffled_nx)

print("df_train_nx shape:", df_train_nx.shape) # 15395 x 4
print("df_val_nx shape:", df_val_nx.shape) # 3299 x 4
print("df_test_nx shape:", df_test_nx.shape) # 3300 x 4
print(df_test_nx.head())
#endregion



######### Gaining more information about data #########

#region Occurrences of tokens: how many in total, how many unique, how often do unique tokens appear?
#region all tokens lists
# Dataframes in dictionary
dataframes = {
    'train_nx': df_train_nx,
    'val_nx': df_val_nx,
    'test_nx': df_test_nx
}
def aggregate_tokens(dataframe):
    return [token for sublist in dataframe['tok_signs'] for token in sublist]

# Place to store dataframes in: all_tokens dataframes
all_tokens = {}

for name, dataframe in dataframes.items():
    all_tokens[f'all_tokens_{name}'] = aggregate_tokens(dataframe)

# Display the aggregated tokens to verify
for name, tokens in all_tokens.items():
    print(f"Token length for df_{name}:")
    print(len(tokens))
    print()

for name, token in all_tokens.items():
    print(f'Length of the tokens in: {name}: {len(token)}')
    # Length of the tokens in: all_tokens_train_nx: 1824260
    # Length of the tokens in: all_tokens_val_nx: 391434
    # Length of the tokens in: all_tokens_test_nx: 403357
#endregion

#region unique tokens list
# Create dataframes with unique token counts
from collections import Counter

unique_token_counts = {}

for name, tokens in all_tokens.items():
    token_counts = Counter(tokens)
    un_tok = pd.DataFrame(token_counts.items(), columns=['token', 'count'])
    unique_name = name.replace('all_tokens_', '')
    unique_token_counts[f'unique_tok_counts_{unique_name}'] = un_tok

# Display the unique token count dataframes to verify
for name, un_tok in unique_token_counts.items():
    print(f"\nUnique token counts for {name}:{len(un_tok['count'])}" )
# Unique token counts for unique_tok_counts_train_nx:4926
# Unique token counts for unique_tok_counts_val_nx:1831
# Unique token counts for unique_tok_counts_test_nx:1857
#endregion

#region How many of the unique tokens appear once?
for name, un_tok in unique_token_counts.items():
    print(f"\nNumber of tokens only appearing once for {name}: {len(un_tok[un_tok['count'] == 1])}")
# Number of tokens only appearing once for unique_tok_counts_train_nx: 2988
# Number of tokens only appearing once for unique_tok_counts_val_nx: 1049
# Number of tokens only appearing once for unique_tok_counts_test_nx: 1086
#endregion

#region How many of the unique tokens appear less or equal than three times?
for name, un_tok in unique_token_counts.items():
print(f"\nNumber of tokens appearing less or equal than three times for {name}: {len(un_tok[un_tok['count'] <= 3])}")
# Number of tokens appearing less or equal than three times for unique_tok_counts_train_nx: 3864
# Number of tokens appearing less or equal than three times for unique_tok_counts_val_nx: 1331
# Number of tokens appearing less or equal than three times for unique_tok_counts_test_nx: 1326
#endregion

#region tokens sorted by their Occurency
sorted_unique_token_counts = {}
print(sorted_unique_token_counts)

for name, un_tok in unique_token_counts.items():
    sorted_un_tok = un_tok.sort_values(by='count', ascending=False).reset_index(drop = True)
    sorted_unique_token_counts[name] = sorted_un_tok

# Display the sorted DataFrames to verify
for name, sorted_un_tok in sorted_unique_token_counts.items():
    print(f"\nSorted unique token counts for {name}:")
    print(sorted_un_tok.head())
#endregion

#region Top 15 tokens
top_15_tables = {}

for name, sorted_un_tok in sorted_unique_token_counts.items():
    top_15 = sorted_un_tok.head(15)
    top_15_tables[name] = top_15

# Display the top 15 tokens tables to verify
for name, top_15 in top_15_tables.items():
    print(f"\nTop 15 tokens for {name}:")
    print(top_15)

# region Plot Top 15
import matplotlib.pyplot as plt

# List of DataFrames to be included in the plot
dfs_to_plot = ['unique_tok_counts_train_nx', 'unique_tok_counts_val_nx', 'unique_tok_counts_test_nx']

# Filter the top_15_tables for the DataFrames to be plotted
top_15_tables_nx = {name: top_15_tables[name] for name in dfs_to_plot}

# Create the plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# Plot each table
for ax, (name, top_15) in zip(axes, top_15_tables_nx.items()):
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=top_15.values, colLabels=top_15.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    ax.set_title(f'Top 15 Tokens in {name}', fontweight='bold')

# Add border to each cell and separate columns
for key, cell in table.get_celld().items():
    cell.set_edgecolor('black')
    cell.set_linewidth(1)

# Adjust layout to have gaps between subplots
fig.subplots_adjust(wspace=0.6)  
plt.savefig('plots/top15')
plt.show()
# endregion

#endregion

# region Top 20 tokens
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

top_20_tables = {}

for name, sorted_un_tok in sorted_unique_token_counts.items():
    top_20 = sorted_un_tok.head(20)
    top_20_tables[name] = top_20

# Display the top 20 tokens tables to verify
for name, top_20 in top_20_tables.items():
    print(f"\nTop 20 tokens for {name}:")
    print(top_20)

# region Plot Top 20
# List of DataFrames to be included in the plot
dfs_to_plot = ['unique_tok_counts_train_nx', 'unique_tok_counts_val_nx', 'unique_tok_counts_test_nx']

# Filter the top_20_tables for the DataFrames to be plotted
top_20_tables_nx = {name: top_20_tables[name] for name in dfs_to_plot}

# Create the plot
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Plot each table
for ax, (name, top_20) in zip(axes, top_20_tables_nx.items()):
    ax.axis('tight')
    ax.axis('off')

    # Create the table with larger font size
    table = ax.table(cellText=top_20.values, colLabels=top_20.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)  # Increase the font size for the table values
    table.scale(1.5, 1.5)  # Scale the table to make it larger

    # Set larger title font size
    ax.set_title(f'Top 20 Tokens in {name}', fontweight='bold', fontsize=12)  # Increase title font size

    # Add border to each cell and separate columns
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(1)

# Adjust layout to have gaps between subplots
fig.subplots_adjust(wspace=0.6)  # Adjust wspace to control the space between the tables
plt.savefig('plots/top20.jpg')
plt.show()
# endregion
#endregion

#region Top 50 tokens 
top_50_tables = {}

for name, sorted_un_tok in sorted_unique_token_counts.items():
    top_50 = sorted_un_tok.head(50)
    top_50_tables[name] = top_50

# Display the top 50 tokens tables to verify
for name, top_50 in top_50_tables.items():
    print(f"\nTop 50 tokens for {name}:")
    print(top_50)

# region Plot Top 50 with highlighted common tokens
# List of DataFrames to be included in the plot
dfs_to_plot = ['unique_tok_counts_train_nx', 'unique_tok_counts_val_nx', 'unique_tok_counts_test_nx']

# Define a light yellow color
highlight_color = mcolors.to_rgba('lightyellow')

# Get the set of common tokens in the top 50 tables
common_tokens_50 = set.intersection(
    *[set(top_50_tables[name].head(50)['token']) for name in dfs_to_plot]
)

# Filter the top_50_tables for the DataFrames to be plotted
top_50_tables_nx = {name: top_50_tables[name] for name in dfs_to_plot}

# Create the plot
fig, axes = plt.subplots(1, 3, figsize=(15, 13.5))

# Plot each table
for ax, (name, top_50) in zip(axes, top_50_tables_nx.items()):
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=top_50.values, colLabels=top_50.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8) 
    table.scale(1.2,1.2)
    ax.set_title(f'Top 50 Tokens in {name}', fontweight='bold')

    # Highlight common tokens
    for i, token in enumerate(top_50['token']):
        if token in common_tokens_50:
            table[(i + 1, 0)].set_facecolor(highlight_color)
            table[(i + 1, 1)].set_facecolor(highlight_color)

    # Add border to each cell and separate columns
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(1)

# Adjust layout to have gaps between subplots
fig.subplots_adjust(wspace=0.6)  
plt.show()
# endregion

#region Top 50 tokens without colors
top_50_tables = {}

for name, sorted_un_tok in sorted_unique_token_counts.items():
    top_50 = sorted_un_tok.head(50)
    top_50_tables[name] = top_50

# Display the top 50 tokens tables to verify
for name, top_50 in top_50_tables.items():
    print(f"\nTop 50 tokens for {name}:")
    print(top_50)

# region Plot Top 50 without highlighted common tokens
import matplotlib.pyplot as plt

# List of DataFrames to be included in the plot
dfs_to_plot = ['unique_tok_counts_train_nx', 'unique_tok_counts_val_nx', 'unique_tok_counts_test_nx']

# Filter the top_50_tables for the DataFrames to be plotted
top_50_tables_nx = {name: top_50_tables[name] for name in dfs_to_plot}

# Create the plot
fig, axes = plt.subplots(1, 3, figsize=(15, 13.5))

# Plot each table
for ax, (name, top_50) in zip(axes, top_50_tables_nx.items()):
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=top_50.values, colLabels=top_50.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)  
    table.scale(1.2,1.2)
    ax.set_title(f'Top 50 Tokens in {name}', fontweight='bold')

    # Add border to each cell and separate columns
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(1)

# Adjust layout to have gaps between subplots
fig.subplots_adjust(wspace=0.7)  
plt.savefig('plots/top50_no-col')
plt.show()
# endregion


#endregion
#endregion

#region Plot unique tokens sorted by count: meq 5
# List of DataFrames to be plotted: X included here!
dfs_to_plot = ['unique_tok_counts_train_nx', 'unique_tok_counts_val_nx', 'unique_tok_counts_test_nx']

# Filter and sort the DataFrames for the bar chart: only those who appear at least 5 times
filtered_sorted_counts_meq5 = {name: df[df['count'] >= 5] for name, df in sorted_unique_token_counts.items() if
                                name in dfs_to_plot}

# Create the plot with three rows
fig, axes = plt.subplots(3, 1, figsize=(15, 15))

# Plot each DataFrame as a bar chart
for ax, (name, df) in zip(axes, filtered_sorted_counts_meq5.items()):
    ax.bar(df['token'], df['count'])

    # Set the title with increased font size
    ax.set_title(f'Token Counts in {name}', fontweight='bold', fontsize=18)  

    # Set the labels with increased font size
    ax.set_xlabel('Tokens', fontsize=16)  
    ax.set_ylabel('Count', fontsize=16) 

    # Set tick parameters with increased font size
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Optionally, keep x-axis ticks hidden as before
    ax.set_xticks([])  

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('plots/unique_tok_counts.jpg')
plt.show()
#endregion

#region Plot count frequencies: How often do counts occur?
# List of DataFrames to be plotted: X included here!
dfs_to_plot = ['unique_tok_counts_train_nx', 'unique_tok_counts_val_nx', 'unique_tok_counts_test_nx']

# Create the plot with three rows
fig, axes = plt.subplots(3, 1, figsize=(15, 15))

# Plot each DataFrame as a bar chart
for ax, name in zip(axes, dfs_to_plot):
    token_counts_df = sorted_unique_token_counts[name]

    # Count unique values in the 'count' column for each DataFrame and exclude frequencies that appear only once
    count_frequencies = token_counts_df['count'].value_counts().sort_index()
    filtered_count_frequencies = count_frequencies[count_frequencies > 1]

    # Plot the bar chart
    filtered_count_frequencies.sort_values(ascending=False).plot(kind='bar', ax=ax)

    # Set the title and labels
    ax.set_title(f'Frequency of Token Counts in {name}', fontweight='bold')
    ax.set_xlabel('Count')
    ax.set_ylabel('Frequency')
    #ax.set_yscale('log')  # Optional: Use a logarithmic scale for better visualization if counts vary widely

    # Annotate each bar with the frequency value
    for p in ax.patches:
        ax.annotate(str(int(p.get_height())),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom',
                    xytext=(0, 10),
                    textcoords='offset points')

    # Adjust y-axis limit to add some padding above the highest bar
    ax.set_ylim(0, max(filtered_count_frequencies.values) * 1.2)
    # Adjust x-axis limit to add some padding on the left
    ax.set_xlim(left=filtered_count_frequencies.index.min() - 2.2)

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('plots/counts_of_counts_m1.jpg')
plt.show()
#endregion

#endregion

#region Checking for Duplicates within datasets - none left!

# Check within each dataset
train_duplicates = df_train_nx[df_train_nx['tok_signs'].duplicated()]
print(train_duplicates)
val_duplicates = df_val_nx[df_val_nx['tok_signs'].duplicated()]
print(val_duplicates)
test_duplicates = df_test_nx[df_test_nx['tok_signs'].duplicated()]
print(test_duplicates)

# Check across datasets
combined_df = pd.concat([df_train_nx[['tok_signs']], df_val_nx[['tok_signs']], df_test_nx[['tok_signs']]], keys=['train', 'val', 'test'])
combined_duplicates = combined_df[combined_df.duplicated()]
print(combined_duplicates)
#endregion

#region Letters and Digits

#region Counting letters and digits of ALL and UNIQUE tokens
def count_letters_digits(token):
    letters = sum(c.isalpha() for c in token)
    digits = sum(c.isdigit() for c in token)
    return letters, digits

#region Counting letters and digits of all tokens
letter_digit_counts_all = {}

for name, tokens in all_tokens.items():
    # Calculate the letter and digit counts for each token
    token_counts = [(token, *count_letters_digits(token)) for token in tokens]

    # Create a DataFrame with columns 'token', 'letters', 'digits'
    token_counts_all_df = pd.DataFrame(token_counts, columns=['token', 'letters', 'digits'])

    # Use the appropriate key format for the dictionary
    l_d_count_all_name = name.replace('all_tokens_', 'l_d_count_all_')
    letter_digit_counts_all[l_d_count_all_name] = token_counts_all_df

# Display the letter and digit counts DataFrames to verify
for name, df in letter_digit_counts_all.items():
    print(f"\nLetter and digit counts for {name}:")
    print(df.head())
#endregion

# region Counting letters and digits of unique tokens
letter_digit_counts_unique = {}

for name, token_counts_df in unique_token_counts.items():
    # Calculate the letter and digit counts for each token
    token_counts_unique = [(token, *count_letters_digits(token)) for token in token_counts_df['token']]

    # Create a DataFrame with columns 'token', 'letters', 'digits'
    token_counts_unique_df = pd.DataFrame(token_counts_unique, columns=['token', 'letters', 'digits'])

    # Use the appropriate key format for the dictionary
    l_d_count_unique_name = name.replace('unique_tok_counts_', 'l_d_count_unique_')
    letter_digit_counts_unique[l_d_count_unique_name] = token_counts_unique_df

# Display the letter and digit counts DataFrames to verify
for name, df in letter_digit_counts_unique.items():
    print(f"\nLetter and digit counts for {name}:")
    print(df.head())
# endregion
#endregion

#region Visualizing Letter and Digit Counts

# region for unique tokens
import matplotlib.pyplot as plt

datasets_to_plot = ['l_d_count_unique_train_nx', 'l_d_count_unique_val_nx', 'l_d_count_unique_test_nx']

# Create the plot with 3 rows and 2 columns
fig, axes = plt.subplots(3, 2, figsize=(18, 18))

# Plot each DataFrame's letter and digit counts as bar charts
for i, name in enumerate(datasets_to_plot):
    df = letter_digit_counts_unique[name]

    # Count the frequency of letter counts
    letter_counts = df['letters'].value_counts().sort_index()

    # Count the frequency of digit counts
    digit_counts = df['digits'].value_counts().sort_index()

    # Plot letter counts
    ax_letters = axes[i, 0]
    letter_counts.plot(kind='bar', ax=ax_letters, color='blue')
    ax_letters.set_title(f'Letter Counts in {name}', fontsize=18)
    ax_letters.set_xlabel('Letter Count', fontsize=16)
    ax_letters.set_ylabel('Frequency', fontsize=16)
    ax_letters.tick_params(axis='both', which='major', labelsize=14)

    # Plot digit counts
    ax_digits = axes[i, 1]
    digit_counts.plot(kind='bar', ax=ax_digits, color='green')
    ax_digits.set_title(f'Digit Counts in {name}', fontsize=18)
    ax_digits.set_xlabel('Digit Count', fontsize=16)
    ax_digits.set_ylabel('Frequency', fontsize=16)
    ax_digits.tick_params(axis='both', which='major', labelsize=14)

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('plots/letter_digit_counts_unique-tokens')  # Correct line here
plt.show()
# endregion

#region for all tokens
import matplotlib.pyplot as plt

# Datasets to be visualized
datasets_to_plot = ['l_d_count_all_train_nx', 'l_d_count_all_val_nx', 'l_d_count_all_test_nx']

# Create the plot with 3 rows and 2 columns
fig, axes = plt.subplots(3, 2, figsize=(18, 18))

# Plot each DataFrame's letter and digit counts as bar charts
for i, name in enumerate(datasets_to_plot):
    df = letter_digit_counts_all[name]

    # Count the frequency of letter counts
    letter_counts = df['letters'].value_counts().sort_index()

    # Count the frequency of digit counts
    digit_counts = df['digits'].value_counts().sort_index()

    # Plot letter counts
    ax_letters = axes[i, 0]
    letter_counts.plot(kind='bar', ax=ax_letters, color='blue')
    ax_letters.set_title(f'Letter Counts in {name}', fontsize=18)  
    ax_letters.set_xlabel('Letter Count', fontsize=16)  
    ax_letters.set_ylabel('Frequency', fontsize=16)  
    ax_letters.ticklabel_format(style='plain', axis='y')
    ax_letters.tick_params(axis='both', which='major', labelsize=14) 

    # Plot digit counts
    ax_digits = axes[i, 1]
    digit_counts.plot(kind='bar', ax=ax_digits, color='green')
    ax_digits.set_title(f'Digit Counts in {name}', fontsize=18)  
    ax_digits.set_xlabel('Digit Count', fontsize=16) 
    ax_digits.set_ylabel('Frequency', fontsize=16)  
    ax_digits.ticklabel_format(style='plain', axis='y')
    ax_digits.tick_params(axis='both', which='major', labelsize=14) 

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('letter_digit_counts_all-tokens.jpg')
plt.show()
#region

    #endregion

#endregion

#endregion
#endregion

#region Table of the 20 most frequently occuring tokens, excluding <NEWLINE>, <BOS>, <EOS>, and X
exclude_tokens = ['<NEWLINE>', '<BOS>', '<EOS>', 'X']
filtered_data = unique_codes[~unique_codes['token'].isin(exclude_tokens)]
top_20 = filtered_data.head(20)

# Creating the figure and a subplot
fig, ax = plt.subplots(figsize=(3.5,4))  
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=top_20.values, colLabels=top_20.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.1, 1.1)

# Save the figure
#plt.savefig('table_image.png')
plt.show()
#endregion

#region Frequency of token counts
# Count the frequencies of the counts
count_frequencies = unique_codes['count'].value_counts().sort_index()
count_frequencies_no_ones = count_frequencies[count_frequencies>1]
count_frequencies_m_two = count_frequencies[count_frequencies>2]
count_frequencies_leq_five = count_frequencies[count_frequencies>5]
print(count_frequencies.head())
print(count_frequencies_no_ones.head())

# Plotting the distribution of token counts higher than 2
plt.figure(figsize=(24, 8))
ax = count_frequencies_no_ones.plot(kind='bar')
plt.title('Distribution of Token Counts (Frequency > 1)')
plt.xlabel('Count of Tokens')
plt.ylabel('Frequency')

# Adding frequency numbers above each bar
for p in ax.patches:
    ax.annotate(str(p.get_height()),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom',
                xytext=(0, 10),
                textcoords='offset points')

# Adjust y-axis limit to add some padding above the highest bar
ax.set_ylim(0, max(count_frequencies_no_ones) * 1.1)
ax.set_xlim(left=filtered_count_frequencies.index.min() - 2.5)

plt.show()
#endregion



######### Further Analysis Preparations #########

#region Create Vocabulary and inversed vocabulary without X and NEWLINE
from collections import Counter

# Flatten the list of tokenized signs to create a vocabulary
all_tokens_nx = [token for sublist in df_raw_nx['tok_signs'] for token in sublist]
print(all_tokens_nx[9])

# Count the frequency of each token
token_counts_nx = Counter(all_tokens_nx)

# Create a vocabulary with token to index mapping
# Reserve indices 0-3 for special tokens
vocab_nx = {token: idx for idx, (token, _) in enumerate(token_counts_nx.items(), start=2)}
vocab_nx['<PAD>'] = 0
vocab_nx['<UNK>'] = 1


# Invert the vocabulary dictionary for decoding (if needed)
inv_vocab_nx = {idx: token for token, idx in vocab_nx.items()}
#endregion

#region Convert tokenized data to input IDs and attention masks
def tokens_to_ids(tokens, vocab, max_len=520):
    token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    token_ids = token_ids[:max_len]  # Truncate to max_len
    attention_mask = [1] * len(token_ids)
    padding_length = max_len - len(token_ids)
    token_ids = token_ids + [vocab['<PAD>']] * padding_length
    attention_mask = attention_mask + [0] * padding_length
    return token_ids, attention_mask

print(df_train_nx['attention_mask'][15])

# Apply the function to each tokenized sequence in dataframes without X or NEWLINE
df_train_nx['input_ids'], df_train_nx['attention_mask'] = zip(*df_train_nx['tok_signs'].apply(lambda x: tokens_to_ids(x, vocab_nx)))
df_val_nx['input_ids'], df_val_nx['attention_mask'] = zip(*df_val_nx['tok_signs'].apply(lambda x: tokens_to_ids(x, vocab_nx)))
df_test_nx['input_ids'], df_test_nx['attention_mask'] = zip(*df_test_nx['tok_signs'].apply(lambda x: tokens_to_ids(x, vocab_nx)))

print(df_train_nx.head())
print(df_train_nx.columns)
print(df_train_nx['input_ids'].head())
print(df_train_nx['tok_signs'][4])
print(len(df_train_nx['input_ids'][1]))
print(len(df_train_nx['input_ids'][2]))
print(len(df_train_nx['input_ids'][20]))


#endregion

#region Create PyTorch Datasets and DataLoaders
import torch
from torch.utils.data import Dataset, DataLoader

class TransliterationDataset(Dataset):
    def __init__(self, df):
        self.input_ids = torch.tensor(df['input_ids'].tolist())
        self.attention_mask = torch.tensor(df['attention_mask'].tolist())
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.input_ids[idx]  # Using input_ids as labels for MLM
        }

# Create datasets without X
train_dataset_nx = TransliterationDataset(df_train_nx)
val_dataset_nx = TransliterationDataset(df_val_nx)
test_dataset_nx = TransliterationDataset(df_test_nx)

# Create the dataloaders without X
train_loader_nx = DataLoader(train_dataset_nx, batch_size=24, shuffle=True)
val_loader_nx = DataLoader(val_dataset_nx, batch_size=24, shuffle=True)
test_loader_nx = DataLoader(test_dataset_nx, batch_size=24)
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

import torch
print(torch.cuda.is_available())

import tensorflow as tf
print(tf.test.is_gpu_available())


######### LM Training: BertLMHead #########

#region BertLMHead
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





