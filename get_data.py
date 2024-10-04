import pandas as pd
from utils import process_df, augment_data, distribute_body_fat
import os

def build_sheet_url(doc_id, sheet_id):
    return f'https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv&gid={sheet_id}'

doc_id = '18N2n8Bkdf6hJsZhqbwJXoB-zC90n22l5kzD4blW4HtQ'
sheet_id = '850914214'
sheet_url = build_sheet_url(doc_id, sheet_id)
df = pd.read_csv(sheet_url)

df = df[['Height', 'Weight', 'Front Image', 'Back Image', 'Training Body Fat %', 'Training Muscle Mass', 'Training Bone Mass', 'Training Bone Density', 'Waist', 'Hips', 'Gender', 'Demographic']]

df = df[:57]
df.head()
df = process_df(df)
df = distribute_body_fat(df)

directory = "saved/dataframes"
if not os.path.exists(directory):
    os.makedirs(directory)

df.to_pickle('saved/dataframes/df.pkl')

