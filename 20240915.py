#region Read and get overview over raw data
#JSON file path
import json
file_path = 'data/Akkadian.json'

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
#endregion

#region Tokenizing the data and removing duplicates
def tokenize_signs(signs):
    signs = signs.replace('\n', ' <NEWLINE> ') # Replace Newline with special Token
    tokens = ['<BOS>'] + signs.split() + ['<EOS>'] # BOS, EOS and separation by whitespace
    return tokens

def tokenize_signs_exc_x(signs):
    signs = signs.replace('\n', ' <NEWLINE> ')  # Replace newline with special token
    tokens = signs.split()  # Split signs by whitespace
    tokens = ['<BOS>'] + [token for token in tokens if token != 'X'] + ['<EOS>']  # Filter out 'X' and add BOS, EOS
    return tokens

df_raw_x = df_raw.copy()
df_raw_x['tok_signs'] = df_raw_x['signs'].apply(tokenize_signs)
df_raw_x = df_raw_x.drop_duplicates(subset=['tok_signs'])
print(df_raw_x.head())
print(len(df_raw)) # 22054
print(len(df_raw_x)) # 22025 -> 29 rows removed

df_raw_nx = df_raw.copy()
df_raw_nx['tok_signs'] = df_raw_nx['signs'].apply(tokenize_signs_exc_x)
df_raw_nx = df_raw_nx.drop_duplicates(subset=['tok_signs'])
print(df_raw_nx.head())
print(len(df_raw)) # 22054
print(len(df_raw_nx)) # 22004 -> 50 rows removed.
#endregion

#region Removing uninformative rows
# sets of uninformative tokens
uninformative_tokens_x = {'<BOS>', '<NEWLINE>', 'X', '<EOS>'}
uninformative_tokens_nx = {'<BOS>', '<NEWLINE>', '<EOS>'}

# Function to check if a row contains only uninformative tokens
def is_informative_x(tokens):
    return not all(token in uninformative_tokens_x for token in tokens)

def is_informative_nx(tokens):
    return not all(token in uninformative_tokens_nx for token in tokens)

# Filter rows
df_raw_x = df_raw_x[df_raw_x['tok_signs'].apply(is_informative_x)]
df_raw_nx = df_raw_nx[df_raw_nx['tok_signs'].apply(is_informative_nx)]
print(df_raw_x.head())
print(len(df_raw_x)) # 22007 -> 18 rows were uninformative with X
print(len(df_raw_nx)) #21994 -> 10 rows completely uninformative without X


# Reset the index if needed
df_raw_x.reset_index(drop=True, inplace=True)
df_raw_nx.reset_index(drop=True, inplace=True)
#endregion

#region Implement train- and test split: 0.7 training data, 0.15 validation data, 0.15 test data
random_seed = 42
df_shuffled_x = df_raw_x.sample(frac=1, random_state=random_seed).reset_index(drop=True)
df_shuffled_nx = df_raw_nx.sample(frac=1, random_state = random_seed).reset_index(drop=True)

# Display the shuffled DataFrame
print(df_shuffled_x.head())
print(df_shuffled_nx.head())

def train_val_test_split(df):
    train_split = int(0.7*len(df))
    val_split = int(0.85*len(df))
    df_train = df[:train_split]
    df_val = df[train_split:val_split]
    df_test= df[val_split:]
    return df_train, df_val, df_test

df_train_x, df_val_x, df_test_x = train_val_test_split(df_shuffled_x)
df_train_nx, df_val_nx, df_test_nx = train_val_test_split(df_shuffled_nx)

print("df_train_x shape:", df_train_x.shape) # 15404 x 4
print("df_val_x shape:", df_val_x.shape) # 3301 x 4
print("df_test_x shape:", df_test_x.shape) # 3302 x 4
print("df_train_nx shape:", df_train_nx.shape) # 15395 x 4
print("df_val_nx shape:", df_val_nx.shape) # 3299 x 4
print("df_test_nx shape:", df_test_nx.shape) # 3300 x 4
#endregion

#region Occurrences of tokens: how many in total, how many unique, how often do unique tokens appear?
#region all tokens lists
# Dataframes in dictionary
dataframes = {
    'train_x': df_train_x,
    'train_nx': df_train_nx,
    'val_x': df_val_x,
    'val_nx': df_val_nx,
    'test_x': df_test_x,
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
    # Length of the tokens in: all_tokens_train_x: 2006457
    # Length of the tokens in: all_tokens_train_nx: 1824260
    # Length of the tokens in: all_tokens_val_x: 436820
    # Length of the tokens in: all_tokens_val_nx: 391434
    # Length of the tokens in: all_tokens_test_x: 417331
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
# Unique token counts for unique_tok_counts_train_x:4938
# Unique token counts for unique_tok_counts_train_nx:4926
# Unique token counts for unique_tok_counts_val_x:1864
# Unique token counts for unique_tok_counts_val_nx:1831
# Unique token counts for unique_tok_counts_test_x:1860
# Unique token counts for unique_tok_counts_test_nx:1857
#endregion

#region How many of the unique tokens appear once?
for name, un_tok in unique_token_counts.items():
    print(f"\nNumber of tokens only appearing once for {name}: {len(un_tok[un_tok['count'] == 1])}")
# Number of tokens only appearing once for unique_tok_counts_train_x: 3037
# Number of tokens only appearing once for unique_tok_counts_train_nx: 2988
# Number of tokens only appearing once for unique_tok_counts_val_x: 1066
# Number of tokens only appearing once for unique_tok_counts_val_nx: 1049
# Number of tokens only appearing once for unique_tok_counts_test_x: 1090
# Number of tokens only appearing once for unique_tok_counts_test_nx: 1086
#endregion

#region How many of the unique tokens appear less or equal than three times?
for name, un_tok in unique_token_counts.items():
print(f"\nNumber of tokens appearing less or equal than three times for {name}: {len(un_tok[un_tok['count'] <= 3])}")
# Number of tokens appearing less or equal than three times for unique_tok_counts_train_x: 3864
# Number of tokens appearing less or equal than three times for unique_tok_counts_train_nx: 3864
# Number of tokens appearing less or equal than three times for unique_tok_counts_val_x: 1331
# Number of tokens appearing less or equal than three times for unique_tok_counts_val_nx: 1331
# Number of tokens appearing less or equal than three times for unique_tok_counts_test_x: 1326
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
fig.subplots_adjust(wspace=0.6)  # Adjust wspace to control the space between the tables
plt.savefig('plots/top15')
plt.show()
# endregion

#endregion

#region OLD Top 20 tokens
import matplotlib.colors as mcolors
top_20_tables = {}

for name, sorted_un_tok in sorted_unique_token_counts.items():
    top_20 = sorted_un_tok.head(20)
    top_20_tables[name] = top_20

# Display the top 20 tokens tables to verify
for name, top_20 in top_20_tables.items():
    print(f"\nTop 20 tokens for {name}:")
    print(top_20)

#region Plot Top 20
# List of DataFrames to be included in the plot
dfs_to_plot = ['unique_tok_counts_train_nx', 'unique_tok_counts_val_nx', 'unique_tok_counts_test_nx']

# Define a light yellow color
highlight_color_yellow = mcolors.to_rgba('lightyellow')

# Get the set of common tokens in the top 20 tables
common_tokens = set.intersection(
    *[set(top_20_tables[name].head(20)['token']) for name in dfs_to_plot]
)

# Filter the top_20_tables for the DataFrames to be plotted
top_20_tables_nx = {name: top_20_tables[name] for name in dfs_to_plot}

# Create the plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

# Plot each table
for ax, (name, top_20) in zip(axes, top_20_tables_nx.items()):
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=top_20.values, colLabels=top_20.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    ax.set_title(f'Top 20 Tokens in {name}', fontweight='bold')

    # Highlight common tokens
    for i, token in enumerate(top_20['token']):
        if token in common_tokens:
            table[(i + 1, 0)].set_facecolor(highlight_color_yellow)
            table[(i + 1, 1)].set_facecolor(highlight_color_yellow)

    # Add border to each cell and separate columns
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(1)

# Adjust layout to have gaps between subplots
fig.subplots_adjust(wspace=0.6)  # Adjust wspace to control the space between the tables

plt.show()
#endregion

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
    table.set_fontsize(8)  # Adjust font size to fit larger table
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
fig.subplots_adjust(wspace=0.6)  # Adjust wspace to control the space between the tables

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
    table.set_fontsize(8)  # Adjust font size to fit larger table
    table.scale(1.2,1.2)
    ax.set_title(f'Top 50 Tokens in {name}', fontweight='bold')

    # Add border to each cell and separate columns
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(1)

# Adjust layout to have gaps between subplots
fig.subplots_adjust(wspace=0.7)  # Adjust wspace to control the space between the tables
plt.savefig('plots/top50_no-col')
plt.show()
# endregion


#endregion
#endregion

#region OLD Plot unique tokens sorted by their count: more or equal than 5
# List of DataFrames to be plotted: X included here!
dfs_to_plot = ['unique_tok_counts_train_x', 'unique_tok_counts_val_x', 'unique_tok_counts_test_x']

# Filter and sort the DataFrames for the bar chart: only those who appear at least 5 times
filtered_sorted_counts_meq5 = {name: df[df['count'] >=5] for name, df in sorted_unique_token_counts.items() if name in dfs_to_plot}


# Create the plot with three rows
fig, axes = plt.subplots(3, 1, figsize=(15, 15))

# Plot each DataFrame as a bar chart
for ax, (name, df) in zip(axes, filtered_sorted_counts_meq5.items()):
    ax.bar(df['token'], df['count'])
    ax.set_title(f'Token Counts in {name}', fontweight='bold')
    ax.set_xlabel('Tokens')
    ax.set_ylabel('Count')
    ax.set_xticks([])  # Remove x-axis labels

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('plots/unique_tok_counts_meq5.jpg')
plt.show()
#endregion

#region Plot unique tokens sorted by count: meq 5
# List of DataFrames to be plotted: X included here!
dfs_to_plot = ['unique_tok_counts_train_x', 'unique_tok_counts_val_x', 'unique_tok_counts_test_x']

# Filter and sort the DataFrames for the bar chart: only those who appear at least 5 times
filtered_sorted_counts_meq5 = {name: df[df['count'] >= 5] for name, df in sorted_unique_token_counts.items() if
                                name in dfs_to_plot}

# Create the plot with three rows
fig, axes = plt.subplots(3, 1, figsize=(15, 15))

# Plot each DataFrame as a bar chart
for ax, (name, df) in zip(axes, filtered_sorted_counts_meq5.items()):
    ax.bar(df['token'], df['count'])

    # Set the title with increased font size
    ax.set_title(f'Token Counts in {name}', fontweight='bold', fontsize=18)  # Increase font size for title

    # Set the labels with increased font size
    ax.set_xlabel('Tokens', fontsize=16)  # Increase font size for x-axis label
    ax.set_ylabel('Count', fontsize=16)  # Increase font size for y-axis label

    # Set tick parameters with increased font size
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Optionally, keep x-axis ticks hidden as before
    ax.set_xticks([])  # Remove x-axis labels

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('plots/unique_tok_counts.jpg')
plt.show()
#endregion

#region OLD: Visualizing the token counts of the Top15
import matplotlib.pyplot as plt
n_rows = 2
n_cols = (len(top_15_tables) + 1) // n_rows  # Ensuring enough columns to fit all plots

# Create the plot
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate

# Plot each table
for ax, (name, top_15) in zip(axes, top_15_tables.items()):
    ax.bar(top_15['token'], top_15['count'])
    ax.set_title(f'Top 15 Tokens in {name}')
    ax.set_xlabel('Tokens')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=90)  # Rotate x-axis labels for better readability

# Remove any empty subplots
for i in range(len(top_15_tables), len(axes)):
    fig.delaxes(axes[i])

# Adjust layout and save the plot
plt.tight_layout()
plt.show()
#endregion

#region Plot count frequencies: How often do counts occur?
# List of DataFrames to be plotted: X included here!
dfs_to_plot = ['unique_tok_counts_train_x', 'unique_tok_counts_val_x', 'unique_tok_counts_test_x']

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
    
#region OLD ... for the unique tokens
# Datasets to be visualized
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
    ax_letters.set_title(f'Letter Counts in {name}')
    ax_letters.set_xlabel('Letter Count')
    ax_letters.set_ylabel('Frequency')

    # Plot digit counts
    ax_digits = axes[i, 1]
    digit_counts.plot(kind='bar', ax=ax_digits, color='green')
    ax_digits.set_title(f'Digit Counts in {name}')
    ax_digits.set_xlabel('Digit Count')
    ax_digits.set_ylabel('Frequency')

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('letter_digit_counts_unique-tokens.jpg')
plt.show()
#endregion

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


# region OLD ... for all tokens
# Datasets to be visualized
import matplotlib.pyplot as plt

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
    ax_letters.set_title(f'Letter Counts in {name}')
    ax_letters.set_xlabel('Letter Count')
    ax_letters.set_ylabel('Frequency')
    ax_letters.ticklabel_format(style='plain', axis='y')

    # Plot digit counts
    ax_digits = axes[i, 1]
    digit_counts.plot(kind='bar', ax=ax_digits, color='green')
    ax_digits.set_title(f'Digit Counts in {name}')
    ax_digits.set_xlabel('Digit Count')
    ax_digits.set_ylabel('Frequency')
    ax_digits.ticklabel_format(style='plain', axis='y')

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('letter_digit_counts_all-tokens.jpg')
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
    ax_letters.set_title(f'Letter Counts in {name}', fontsize=18)  # Increased font size for title
    ax_letters.set_xlabel('Letter Count', fontsize=16)  # Increased font size for x-axis label
    ax_letters.set_ylabel('Frequency', fontsize=16)  # Increased font size for y-axis label
    ax_letters.ticklabel_format(style='plain', axis='y')
    ax_letters.tick_params(axis='both', which='major', labelsize=14)  # Increased font size for tick labels

    # Plot digit counts
    ax_digits = axes[i, 1]
    digit_counts.plot(kind='bar', ax=ax_digits, color='green')
    ax_digits.set_title(f'Digit Counts in {name}', fontsize=18)  # Increased font size for title
    ax_digits.set_xlabel('Digit Count', fontsize=16)  # Increased font size for x-axis label
    ax_digits.set_ylabel('Frequency', fontsize=16)  # Increased font size for y-axis label
    ax_digits.ticklabel_format(style='plain', axis='y')
    ax_digits.tick_params(axis='both', which='major', labelsize=14)  # Increased font size for tick labels

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('letter_digit_counts_all-tokens.jpg')
plt.show()
#region

    #endregion

#endregion

#endregion
#endregion



#region OLD: Plotting the counts of letters
import matplotlib.pyplot as plt
plt.figure(figsize=(24, 10))

plt.subplot(2, 2, 1)
all_counts_df['letters'].value_counts().sort_index().plot(kind='bar')
plt.title('Letter counts in all tokens')
plt.xlabel('Letter count per token')
plt.ylabel('Frequency')
plt.ticklabel_format(style='plain', axis='y')

# Plotting the counts of digits
plt.subplot(2, 2, 2)
all_counts_df['digits'].value_counts().sort_index().plot(kind='bar')
plt.title('Digit counts in all tokens')
plt.xlabel('Digit count per token')
plt.ylabel('Frequency')
plt.ticklabel_format(style='plain', axis='y')

plt.subplot(2, 2, 3)
unique_counts_df['letters'].value_counts().sort_index().plot(kind='bar')
plt.title('Letter counts in unique tokens')
plt.xlabel('Letter count per token')
plt.ylabel('Frequency')
plt.ticklabel_format(style='plain', axis='y')

# Plotting the counts of digits
plt.subplot(2, 2, 4)
unique_counts_df['digits'].value_counts().sort_index().plot(kind='bar')
plt.title('Digit counts in unique tokens')
plt.xlabel('Digit count per token')
plt.ylabel('Frequency')
plt.ticklabel_format(style='plain', axis='y')

plt.tight_layout()
#plt.savefig('letters_digit_counts.png')
plt.show()
#endregion

#region OLD Print tokens that seem to have an odd pattern: First filter, then print
## NOT included: 3 digits or letters (most of them have that pattern), only 1 letter (X), 7 letters (NEWLINE),
odd_tokens = unique_counts_df[
    (unique_counts_df['digits'].isin([1, 2])) |
    (unique_counts_df['letters'].isin([0, 4, 6]))
]
odd_tokens.head()
print(odd_tokens[['token', 'letters', 'digits']])
#endregion

#region Table of the 20 most frequently occuring tokens, excluding <NEWLINE>, <BOS>, <EOS>, and X
exclude_tokens = ['<NEWLINE>', '<BOS>', '<EOS>', 'X']
filtered_data = unique_codes[~unique_codes['token'].isin(exclude_tokens)]
top_20 = filtered_data.head(20)

# Creating the figure and a subplot
fig, ax = plt.subplots(figsize=(3.5,4))  # Adjust the size as needed
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

#region NEW: Create Vocabulary and inversed vocabulary with X
from collections import Counter

# Flatten the list of tokenized signs to create a vocabulary
all_tokens_x = [token for sublist in df_train_x['tok_signs'] for token in sublist]
print(all_tokens_x[9])

# Count the frequency of each token
token_counts_x = Counter(all_tokens_x)

# Create a vocabulary with token to index mapping
# Reserve indices 0-3 for special tokens
vocab_x = {token: idx for idx, (token, _) in enumerate(token_counts_x.items(), start=2)}
vocab_x['<PAD>'] = 0
vocab_x['<UNK>'] = 1


# Invert the vocabulary dictionary for decoding (if needed)
inv_vocab_x = {idx: token for token, idx in vocab_x.items()}
#endregion

#region NEW: Create Vocabulary and inversed vocabulary without X
from collections import Counter

# Flatten the list of tokenized signs to create a vocabulary
all_tokens_nx = [token for sublist in df_train_nx['tok_signs'] for token in sublist]
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
def tokens_to_ids(tokens, vocab, max_len=512):
    token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    token_ids = token_ids[:max_len]  # Truncate to max_len
    attention_mask = [1] * len(token_ids)
    padding_length = max_len - len(token_ids)
    token_ids = token_ids + [vocab['<PAD>']] * padding_length
    attention_mask = attention_mask + [0] * padding_length
    return token_ids, attention_mask

# Apply the function to each tokenized sequence in the dataframes with X
df_train_x['input_ids'], df_train_x['attention_mask'] = zip(*df_train_x['tok_signs'].apply(lambda x: tokens_to_ids(x, vocab_x)))
df_val_x['input_ids'], df_val_x['attention_mask'] = zip(*df_val_x['tok_signs'].apply(lambda x: tokens_to_ids(x, vocab_x)))
df_test_x['input_ids'], df_test_x['attention_mask'] = zip(*df_test_x['tok_signs'].apply(lambda x: tokens_to_ids(x, vocab_x)))

# Without X
df_train_nx['input_ids'], df_train_nx['attention_mask'] = zip(*df_train_nx['tok_signs'].apply(lambda x: tokens_to_ids(x, vocab_nx)))
df_val_nx['input_ids'], df_val_nx['attention_mask'] = zip(*df_val_nx['tok_signs'].apply(lambda x: tokens_to_ids(x, vocab_nx)))
df_test_nx['input_ids'], df_test_nx['attention_mask'] = zip(*df_test_nx['tok_signs'].apply(lambda x: tokens_to_ids(x, vocab_nx)))


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
# Create the datasets with X
train_dataset_x = TransliterationDataset(df_train_x)
val_dataset_x = TransliterationDataset(df_val_x)
test_dataset_x = TransliterationDataset(df_test_x)

# Create datasets without X
train_dataset_nx = TransliterationDataset(df_train_nx)
val_dataset_nx = TransliterationDataset(df_val_nx)
test_dataset_nx = TransliterationDataset(df_test_nx)

# Create the dataloaders with X
train_loader_x = DataLoader(train_dataset_x, batch_size=32, shuffle=True)
val_loader_x = DataLoader(val_dataset_x, batch_size=32, shuffle=True)
test_loader_x = DataLoader(test_dataset_x, batch_size=32)

# Create the dataloaders without X
train_loader_x = DataLoader(train_dataset_x, batch_size=32, shuffle=True)
val_loader_x = DataLoader(val_dataset_x, batch_size=32, shuffle=True)
test_loader_x = DataLoader(test_dataset_x, batch_size=32)
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

#region OLD: Define BERT Configuration
from transformers import BertConfig, BertForMaskedLM, Trainer, TrainingArguments

# Define the BERT configuration
config = BertConfig(
    vocab_size=len(vocab),
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
)

# Initialize the model
model_config = BertForMaskedLM(config)
#endregion

#region OLD: Define training arguments and initialize Trainer

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Initialize the Trainer with X
trainer_x = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_x,
    eval_dataset=val_dataset_x,
)

# Initialize the Trainer without X
trainer_nx = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_nx,
    eval_dataset=val_dataset_nx,
)
#endregion

#region OLD: Train the model
trainer_x.train()

print('Unnecessary commit')
#endregion

#region OLD: Bert .pretrained
from transformers import BertForMaskedLM, Trainer, TrainingArguments, BertTokenizer

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
model = BertForMaskedLM.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

#region Define training arguments and initialize Trainer

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Initialize the Trainer with X
trainer_x = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_x,
    eval_dataset=val_dataset_x,
)

# Initialize the Trainer without X
trainer_nx = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_nx,
    eval_dataset=val_dataset_nx,
)
#endregion

# Train the model
trainer_x.train()
trainer_nx.train()

print('Unnecessary commit')
#endregion

#region OLD BertLMHead
from transformers import BertLMHeadModel, Trainer, TrainingArguments, BertTokenizer

# Load the pre-trained BERT model and tokenizer
config = BertConfig.from_pretrained('bert-base-uncased')
config.is_decoder = True
model = BertLMHeadModel.from_pretrained('bert-base-uncased', config=config)
tokenizer = BertTokenizer.from_pretrained(model)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=20,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=300,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Initialize the Trainer without X
trainer_nx = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_nx,
    eval_dataset=val_dataset_nx,
    callbacks=[PerplexityCallback()]
)

#endregion BertLMHead

#region OLD Train BertLMHead
# Train the model
trainer_nx.train()

print('Unnecessary commit')
#endregion BertLMHead

import torch
print(torch.cuda.is_available())

import tensorflow as tf
print(tf.test.is_gpu_available())

#region NEW BertLMHead
#### BERTLMHEAD

import wandb
from transformers import BertLMHeadModel, Trainer, TrainingArguments, BertTokenizer, BertConfig
from transformers.integrations import WandbCallback

# Initialize Weights and Biases
wandb.init(project="master_thesis")

# Load the pre-trained BERT model and tokenizer
config = BertConfig.from_pretrained('bert-base-uncased')
config.is_decoder = True
model = BertLMHeadModel.from_pretrained('bert-base-uncased', config=config)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=20,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=300,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    report_to="wandb"  # Enable reporting to W&B
)

# Initialize the Trainer
trainer_nx = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_nx,
    eval_dataset=val_dataset_nx,
    callbacks=[WandbCallback(), PerplexityCallback()]  

# Train the model
trainer_nx.train()

# Evaluate the model
test_result_nx = trainer_nx.evaluate(eval_dataset=test_dataset_nx)
print("Test Loss: ", test_result_nx['eval_loss'])
#endregion