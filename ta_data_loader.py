# Custom dataset loader for textattack
import numpy as np
import pandas as pd

import textattack

# Load a dataset
data_path = "data/data300k/test.csv"
df = pd.read_csv(data_path)
# there are some rows with label = 'label', we need to remove them
df = df[df['label'] != 'label']
# convert label to int
df['label'] = df['label'].astype(int)

# transform df into a list of tuples [(text, label), ...]
data = list(zip(df['text'].tolist(), df['label'].tolist()))
dataset = textattack.datasets.Dataset(data)
