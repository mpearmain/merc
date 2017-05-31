import pandas as pd
import numpy as np


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