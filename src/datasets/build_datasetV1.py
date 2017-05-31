import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from src.features.build_outlier_features import recode_outlier

#from src.features.build_ordinal_features import
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

"""
A script to build different styles of dataset based on the files in ./src/features.

Scripts should be able to run from ./data/raw and final results should be stored in ./data/processed

For example, we may want to create a dataset based only on transformed missing values, or the combination of missing 
values, and floats.

Simply pick the functions required to 'build' a dataset and run models on these different datasets.

"""

DATASET_NAME = 'V1_base'

# Read the base data
print('Loading Train data ...')
train_df = pd.read_csv('./data/raw/train.csv')
print('Loaded')

print("Recoding Outliers")

train_df['y'] = recode_outlier(train_df['y'], ulimit=190)

print("Recoding")

