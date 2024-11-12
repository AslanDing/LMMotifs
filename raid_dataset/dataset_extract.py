import numpy as np
import pandas as pd
from raid.utils import load_data

train_set = '../data/raid/data/train_none.csv'
train_data = pd.read_csv(train_set)
print(train_data.columns)
print(train_data.iloc[1])
print()

unique_columns = train_data['model'].unique()
print(unique_columns)

# subset_dict = {}
train_df_lists = []
val_df_lists = []
test_df_lists = []


subset_current = train_data[train_data['model'] == 'human']
shuffled_df = subset_current.sample(frac=1, random_state=1)
shuffled_df = shuffled_df.reset_index(drop=True)
train_df = shuffled_df.iloc[:1000*(len(unique_columns)-1)]
test_df = shuffled_df.iloc[1000*(len(unique_columns)-1):1000*(len(unique_columns)-1)+1000*(len(unique_columns)-1)]
val_df = shuffled_df.iloc[2000*(len(unique_columns)-1):2000*(len(unique_columns)-1)+100*(len(unique_columns)-1)]
train_df_lists.append(train_df)
test_df_lists.append(test_df)
val_df_lists.append(val_df)

# subset_dict['human'] = shuffled_df
for name in unique_columns[1:]:
    subset_current = train_data[train_data['model'] == name]

    shuffled_df = subset_current.sample(frac=1, random_state=1)
    shuffled_df = shuffled_df.reset_index(drop=True)
    train_df = shuffled_df.iloc[:1000]
    test_df = shuffled_df.iloc[1000:2000]
    val_df = shuffled_df.iloc[2000:2100]

    # subset_dict[name] = shuffled_df
    train_df_lists.append(train_df)
    val_df_lists.append(val_df)
    test_df_lists.append(test_df)

train_frame  = pd.concat(train_df_lists, axis=0)
train_frame.to_csv('../data/raid/subset/train_subset.csv')
test_frame  = pd.concat(test_df_lists, axis=0)
test_frame.to_csv('../data/raid/subset/test_subset.csv')
val_frame  = pd.concat(val_df_lists, axis=0)
val_frame.to_csv('../data/raid/subset/val_subset.csv')

