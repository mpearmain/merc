import fastparquet
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

"""
Rip of forum base data - to use with models3 - feel weird to use anything but trees on encoded data but there we go
"""

BUILD_NAME = '_build_datasetV3'
NOT_BINARY_COLS = ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]
CONSTANT_COLS = ['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X347']

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

# process columns, apply LabelEncoder to categorical features
for c in train_df.columns:
    if train_df[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train_df[c].values) + list(test_df[c].values)) 
        train_df[c] = lbl.transform(list(train_df[c].values))
        test_df[c] = lbl.transform(list(test_df[c].values))

# shape        
print('Shape train_df: {}\nShape test_df: {}'.format(train_df.shape, test_df.shape))

###########################################################
from sklearn.decomposition import PCA, FastICA

n_comp = 10
# PCA
pca = PCA(n_components=n_comp, random_state=42)
pca2_results_train_df = pca.fit_transform(train_df)
pca2_results_test_df = pca.transform(test_df)

# ICA
ica = FastICA(n_components=n_comp, random_state=42)
ica2_results_train_df = ica.fit_transform(train_df)
ica2_results_test_df = ica.transform(test_df)

# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    train_df['pca_' + str(i)] = pca2_results_train_df[:, i - 1]
    test_df['pca_' + str(i)] = pca2_results_test_df[:, i - 1]

    train_df['ica_' + str(i)] = ica2_results_train_df[:, i - 1]
    test_df['ica_' + str(i)] = ica2_results_test_df[:, i - 1]

# Now split into train and test and save the output of the processed dataset.
train_df['ID'] = id_train
train_df['y'] = y_train
test_df['ID'] = id_test
print('Shape train_df: {}\nShape test_df: {}'.format(train_df.shape, test_df.shape))
print('Writing Parquets')
# store
fastparquet.write('./data/processed/xtrain' + BUILD_NAME + '.parq', train_df, write_index=False)
fastparquet.write('./data/processed/xtest' + BUILD_NAME + '.parq', test_df, write_index=False)
print('Finished')
