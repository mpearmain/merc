import fastparquet
from os.path import join
from functools import reduce
from itertools import combinations, chain
import pandas as pd
import numpy as np
from glob import glob

# from src.features.build_ordinal_features import
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

"""
Pick up the stacking data and the base data for additional feature eng.

"""

BUILD_NAME = '_build_dataset_metalevel2V1'
BASE_COLS = ["y", "ID", "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]
METAS_PATH = r"./data/processed/metalvl1/"

# Read the base data
print('Loading data ...')
train_df = pd.read_csv('./data/raw/train.csv', usecols=BASE_COLS)
test_df = pd.read_csv('./data/raw/test.csv', usecols=list(set(BASE_COLS) - set(["y"])))
print('Loaded')

print("Remove Outliers in y")
train_df = train_df[train_df.y < 180]
print("Removed")

# Read the base data
print('Loading metalevel1 train data ...')

all_train = glob(join(METAS_PATH, "xtrain*"))
train_lvl1 = [fastparquet.ParquetFile(f).to_pandas() for f in all_train] + [train_df]
train_df = reduce(lambda left, right: pd.merge(left, right, on=['ID', 'y']), train_lvl1)

all_test = glob(join(METAS_PATH, "xtest*"))
test_lvl1 = [fastparquet.ParquetFile(f).to_pandas() for f in all_test] + [test_df]
test_df = reduce(lambda left, right: pd.merge(left, right, on='ID'), test_lvl1)

del test_lvl1, train_lvl1, all_test, all_train
print('Loaded')

y_train_df = train_df['y'].values
id_train_df = train_df['ID'].values
id_test_df = test_df['ID'].values

print("Remove constant cols")
train_df = train_df.drop(['ID', 'y'], axis=1)
test_df = test_df.drop(['ID'], axis=1)
print("Removed")



###########################################################

# Create interaction features
interactions2way = list(set(list(train_df)) - set(BASE_COLS))
interactions2way_list = list(combinations(interactions2way, 2))
for A, B in interactions2way_list:
    feat = "_".join([A, B])
    train_df[feat] = abs(train_df[A] - train_df[B])
    test_df[feat] = abs(test_df[A] - test_df[B])

# Now split into train_df and test_df and save the output of the processed dataset.
train_df['ID'] = id_train_df
train_df['y'] = y_train_df
test_df['ID'] = id_test_df

print('Writing Parquets')
# store
fastparquet.write('./data/processed/metalvl2/xtrain' + BUILD_NAME + '.parq', train_df, write_index=False)
fastparquet.write('./data/processed/metalvl2/xtest' + BUILD_NAME + '.parq', test_df, write_index=False)
print('Finished')
