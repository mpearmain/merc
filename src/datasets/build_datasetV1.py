import fastparquet
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from src.features.build_rowwise_binary_features import binary_counts, binary_hashmap

# from src.features.build_ordinal_features import
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

"""
A script to build different styles of dataset based on the files in ./src/features.

Scripts should be able to run from ./data/raw and final results should be stored in ./data/processed

For example, we may want to create a dataset based only on transformed missing values, or the combination of missing 
values, and floats.

Simply pick the functions required to 'build' a dataset and run models on these different datasets.

"""

BUILD_NAME = '_build_datasetV1'
NOT_BINARY_COLS = ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]
CONSTANT_COLS = ['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X347']
# https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/34137 - Complementary cols
COMPLEMENTARY_COLS = ['X29', 'X52', 'X54', 'X76', 'X128', 'X142', 'X186', 'X204', 'X232', 'X263', 'X156']

# Read the base data
print('Loading data ...')
train_df = pd.read_csv('./data/raw/train.csv')
test_df = pd.read_csv('./data/raw/test.csv')
print('Loaded')

print("Remove Outliers in y")
train_df = train_df[train_df.y < 180]
print("Removed")

y_train = train_df['y'].values
id_train = train_df['ID'].values
id_test = test_df['ID'].values

print("Remove constant cols")
train_df = train_df.drop(['ID', 'y'] + CONSTANT_COLS, axis=1)
test_df = test_df.drop(['ID'] + CONSTANT_COLS, axis=1)
print("Removed")

ntrain = train_df.shape[0]
print("Joining train and test for row-wise transforms")
df = pd.concat([train_df, test_df], axis=0)

###########################################################

### Working on binaries Row-wise.
print("Building Binary counts values")
df['zero_counts'] = binary_counts(df[df.columns.difference(NOT_BINARY_COLS)], 0)
df['one_counts'] = binary_counts(df[df.columns.difference(NOT_BINARY_COLS)], 1)
df['binary_prop'] = df['one_counts'] / df['zero_counts']
df['binary_ones_pct'] = df['one_counts'] / (df['one_counts'] + df['zero_counts'])
df['binary_zeros_pct'] = df['zero_counts'] / (df['one_counts'] + df['zero_counts'])
df['binary_hashmap'] = binary_hashmap(df[df.columns.difference(NOT_BINARY_COLS)], low_count=4)

# For Tree based models only - Lets recode non-binary cols to be used in base models.
print("Recoding object values")
for col in NOT_BINARY_COLS + ['binary_hashmap']:
    df[col] = pd.factorize(df[col])[0]

# Now split into train and test and save the output of the processed dataset.
xtrain = df[:ntrain].copy()
xtest = df[ntrain:].copy()
xtrain['ID'] = id_train
xtrain['y'] = y_train
xtest['ID'] = id_test

print('Writing Parquets')
# store
fastparquet.write('./data/processed/xtrain' + BUILD_NAME + '.parq', xtrain, write_index=False)
fastparquet.write('./data/processed/xtest' + BUILD_NAME + '.parq', xtest, write_index=False)
print('Finished')
