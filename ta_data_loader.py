# Custom dataset loader for textattack
import pandas as pd

import textattack

# Load a dataset
data_path = "data/data300k-with-3stars/test.csv"
df = pd.read_csv(data_path)

# transform df into a list of tuples [(text, label), ...]
data = list(zip(df['text'].tolist(), df['label'].tolist()))
dataset = textattack.datasets.Dataset(data)
