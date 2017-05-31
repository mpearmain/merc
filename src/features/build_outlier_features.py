


def recode_outlier(data_col, ulimit=180):
    data_col.loc[data_col['y'] > ulimit] = ulimit
    return data_col
