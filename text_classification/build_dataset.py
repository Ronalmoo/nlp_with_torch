import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# loading dataset
cwd = Path.cwd()
filepath = cwd / 'data/ratings_train.txt'
dataset = pd.read_csv(filepath, sep='\t').loc[:, ['document', 'label']]
dataset = dataset.loc[dataset['document'].isna().apply(lambda x: not x), :]
tr, val = train_test_split(dataset, test_size=0.2, random_state=42)

tr.to_csv(cwd/'data'/'train.txt', sep='\t', index=False)
val.to_csv(cwd/'data'/'validation.txt', sep='\t', index=False)

tst_filepath = cwd / 'data/ratings_test.txt'
test = pd.read_csv(tst_filepath, sep='\t').loc[:, ['document', 'label']]
test = test.loc[test['document'].isna().apply(lambda x: not x), :]
test.to_csv(cwd/'data'/'test.txt', sep='\t', index=False)