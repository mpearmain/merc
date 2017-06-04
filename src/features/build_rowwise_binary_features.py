import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FastICA


def binary_counts(df, value):
    counts = (df == value).sum(axis=1)
    return counts


def binary_hashmap(df, low_count=5):
    # Find a bitmap hash of values
    df['hashes'] = df.apply(lambda x: hash("_".join([c[0] for c in x.items() if c[1] == 1])), axis=1)
    # Find the count of those hashes and recode if they are to low
    df['hashes'].loc[df.groupby('hashes')['hashes'].transform('count') < low_count] = 'LOW_COUNT'
    # Finally recount after recoding low values.
    hashes = df.groupby('hashes')['hashes'].transform('count')
    return hashes


def decomp_features(train, test, n_comp=10):
    # PCA
    train_cols = train.columns.tolist()
    test_cols = test.columns.tolist()

    pca = PCA(n_components=n_comp, random_state=42)
    pca2_results_train = pca.fit_transform(train)
    pca2_results_test = pca.transform(test)

    # ICA
    ica = FastICA(n_components=n_comp, random_state=42, max_iter=10000)
    ica2_results_train = ica.fit_transform(train)
    ica2_results_test = ica.transform(test)

    # Append decomposition components to datasets
    for i in range(1, n_comp + 1):
        train['pca_' + str(i)] = pca2_results_train[:, i - 1]
        test['pca_' + str(i)] = pca2_results_test[:, i - 1]

        train['ica_' + str(i)] = ica2_results_train[:, i - 1]
        test['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train = train.drop(train_cols, axis=1)
    test = test.drop(test_cols, axis=1)

    return train, test


def encode_dim_reduce(train, test, categorical_cols, encode_type, n_comp=10):
    return
