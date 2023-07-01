# %%
import numpy as np 
import pandas as pd
import os

# %%
test_data = pd.read_csv('data/test/000000000000.csv', sep='\t')
print(test_data.shape)

# %%
train = []
files = os.listdir('data/train/')
for f in files:
    df = pd.read_csv('data/train/{}'.format(f), sep='\t')
    train.append(df)

# %%
train_data = pd.concat(train)
print(train_data.shape)

# %%
train_data.to_csv('./data/train.csv', index=False, sep='\t')
print(train_data.shape)

# %%



