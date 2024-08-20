import pandas as pd
from utils import process_df, augment_data, distribute_body_fat
import os

def build_sheet_url(doc_id, sheet_id):
    return f'https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv&gid={sheet_id}'

doc_id = '18N2n8Bkdf6hJsZhqbwJXoB-zC90n22l5kzD4blW4HtQ'
# sheet_id = '1996632023'
sheet_id = '850914214'
sheet_url = build_sheet_url(doc_id, sheet_id)
df = pd.read_csv(sheet_url)

df = df[['Height', 'Weight', 'Front Image', 'Back Image', 'Training Body Fat %', 'Waist', 'Hips']]

df = process_df(df)
df = distribute_body_fat(df)

directory = "saved/dataframes"
if not os.path.exists(directory):
    os.makedirs(directory)

df.to_pickle('saved/dataframes/df.pkl')

df.head()
len(df)
# Rows with missing values seperate the train/val/test splits
# separation_indices = df[df['Height'].isnull()].index

# train_df = process_df(df.iloc[0:separation_indices[0]])
# val_df = process_df(df.iloc[separation_indices[0]+1:separation_indices[1]])
# test_df = process_df(df.iloc[separation_indices[1]+1:separation_indices[2]])

# train_df = distribute_body_fat(train_df)

# print(f"Before Augmentation: Train: {len(train_df)} Val: {len(val_df)} Test: {len(test_df)}")

# train_df = augment_data(train_df, 300)
# val_df = augment_data(val_df, 8)

# print(f"After Augmentation: Train: {len(train_df)} Val: {len(val_df)} Test: {len(test_df)}")

# Check if the directory exists, if not, create it
# if not os.path.exists('saved/dataframes'):
#     os.makedirs('saved/dataframes')
# train_df.to_pickle('saved/dataframes/train_df.pkl')
# val_df.to_pickle('saved/dataframes/val_df.pkl')
# test_df.to_pickle('saved/dataframes/test_df.pkl')
